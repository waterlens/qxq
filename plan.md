### Recommended semantics

- A type declaration is like a normal binding but specific to types. It creates
  a first-class runtime type value and evaluates as `let`.
- Field layout follows declaration order: `year=0`, `month=1`, `day=2`.
- Positional aliases map `a=0`, `b=1`, `c=2`, as required by the test.
  Declared names map to the same offsets.
- In a constructor, labeled arguments select their declared slot; unlabeled
  arguments fill the next unassigned field slot.
- Reject unknown, duplicate, missing, and excess fields.
- Evaluate initializer expressions in source order, even when labels rearrange
  storage.
- Require the first source-level method parameter to be `self`.
- At runtime, `self` is the method's first parameter and is passed in `r0`.
  Explicit parameters begin at `r1`.
- Fields and methods occupy one member namespace. Runtime member lookup maps a
  member-name string to a value slot without attaching a field or method kind to
  the entry.
- `receiver.method(args)` evaluates the receiver once, evaluates the explicit
  arguments in source order, looks up the member closure, and calls it with the
  receiver in `r0`.

### Parser and AST

Add representations along these lines:

```text
StructDecl { name, fields, methods, info }
MethodDecl { name, params, body, info }
Initializer { paired, designation, ordered }
Member { receiver, member, info }
MemberApply { receiver, member, args, info }
```

Parsing changes:

- Add `type` and `struct` as keywords.
- Parse `.member` as member access and `.member(...)` as member application.
- Parse comma-separated constructor braces specially, so `year = 2026` is an
  initializer label rather than a general `=` expression.
- Keep member names and initializer labels out of ordinary free-variable
  collection.
- Record the first `MethodDecl` parameter as the ordinary binding for `self`;
  code generation assigns that binding to `r0`.
- Make all methods declared by the same type visible while processing each
  method body. Resolve such a name as a compiler-only `MemberRef` binding.

Then `day.get_year()` naturally parses as:

```text
MemberApply(day, get_year, [])
```

### Compile-time, image, and runtime descriptions

Keep a compile-time struct description for:

- Declared fields and their stable slots.
- Positional aliases and declared-name aliases.
- Declared methods and their stable slots.
- Constructor designation validation.
- Resolving sibling method names as `MemberRef`.

Use one stable member layout. Declared fields occupy their declaration-order
slots, followed by methods in declaration order. Multiple names may map to one
field slot, as with `year` and `a`.

Store the immutable structural information in a metadata region shared by the
whole bytecode image:

```text
bytecode image:
  type descriptions[]

type description:
  declared field count
  total logical slot count
  member entries = (member-name string, logical slot)
  method count and declaration order
```

Every member entry has the same representation. The field count and method
order give `WObj` the construction recipe: constructor values fill the leading
field slots, and method closures fill the trailing method slots.

The runtime values have these logical shapes:

```text
runtime type value:
  type-description reference
  ordinary method closures in declaration order

struct instance:
  runtime type-value pointer
  member values[logical slot]
```

The type-description reference is an immutable image-owned handle. Runtime type
values are first-class GC objects, so separate executions of one type
declaration can share the description while carrying different ordinary method
closures.

Reserve object tags for runtime type values and struct instances. Member lookup
follows the instance's runtime type value to its image description and returns
the logical slot associated with the requested string. The member operand in
bytecode is a string constant, rather than a physical slot offset.

### Type-declaration lowering

A `StructDecl` lowers like a `Bind`:

1. Establish the compile-time struct description and all of its `MemberRef`
   bindings before compiling any method body.
2. Add its immutable type description to the bytecode image and retain the
   resulting description index.
3. Compile each method as an ordinary thunk template. Its first parameter is
   `self` in `r0`; its explicit parameters start at `r1`.
4. Preserve each method's ordinary lexical free variables and constant table.
5. When the declaration executes, emit `LoadType` for the image description,
   followed by one existing `Clos` per method in declaration order.
6. Wrap that contiguous region with the runtime type tag. `WObj` consumes the
   description handle and method closures and creates the first-class runtime
   type value.
7. Bind the runtime type value to the declared type name and return unit,
   following normal binding behavior.

Method closures created by one execution of the type declaration can be reused
by every instance constructed from that runtime type value. The concrete
receiver is supplied by each `Invoke`.

### Constructor lowering

For a constructor:

1. Resolve and validate every designation against the compile-time struct
   description.
2. Allocate a contiguous register region containing the runtime type binding
   followed by one register per declared field.
3. Put the runtime type value in the first register.
4. Evaluate initializer expressions in source order, writing each result into
   its resolved field-slot register.
5. Wrap the complete region as a struct instance with `WObj`. For the struct
   instance tag, the first register supplies the runtime type value. `WObj`
   reads its image description, allocates the complete member array, copies the
   remaining registers into the leading field slots, and copies the runtime
   type value's method closures into the trailing method slots.

The completed instance is rooted by the destination register before subsequent
allocating operations can run.

### Method code generation

While compiling a method:

- Map `self` to `Slot(0)`.
- Map the first explicit parameter to `Slot(1)`, and continue in order.
- Capture ordinary lexical variables through the existing closure mechanism.
- Lower `self.member` to `LoadField` with the register containing `self` as the
  explicit receiver.
- Lower a sibling `MemberRef` value to `LoadField` using the current `self`.
- Lower a sibling method call to `Invoke` using the current `self`.

A nested closure that uses `self` captures the enclosing method's `Slot(0)`
through the ordinary free-variable mechanism. A `MemberRef` used in that nested
closure lowers through the captured `self` value.

Method references therefore resolve through the current receiver at execution
time. This gives mutually recursive methods access to one another after the
instance member slots have been initialized.

### Member-name operands

`LoadField`, `SetField`, and `Invoke` use the `ABC` encoding. Their member
operand is an 8-bit index into the current thunk's constant table:

```text
@member8 -> string constant
```

Reuse an existing identical string constant where possible. If code generation
cannot assign a referenced member string an 8-bit constant-table index, report
a compile error.

### Object and method bytecode

#### `LoadType`

```text
LoadType rDst, @type
```

`LoadType` uses the `AB` encoding. Its 16-bit index addresses the bytecode
image's type-description region, independently of the current thunk's constant
table. It places a GC-safe image-description handle in `rDst`; `WObj` consumes
that handle when it creates a runtime type value.

Type-declaration code prepares one contiguous region:

```text
LoadType rDst,     @type
Clos     rDst + 1, method0
Clos     rDst + 2, method1
...
WObj     rDst, tag::type, #(1 + method count)
```

For `tag::type`, `WObj` obtains the method count and slot order from the loaded
description, then stores the supplied ordinary closures in the runtime type
value.

#### `LoadFree`

Keep closure free-variable access as a separate operation:

```text
LoadFree rDst, ^freevar
```

Rename the existing `LoadF` spelling to `LoadFree`. It continues to index the
current ordinary closure's capture array.

#### `LoadField`

```text
LoadField rDst, rReceiver, @member8
```

The VM handler:

1. Reads the member-name string from the current thunk's constant table.
2. Loads the receiver from `rReceiver`.
3. Uses the receiver's runtime type value and image description to resolve the
   string to a logical slot.
4. Copies the value in that slot to `rDst`.

#### `SetField`

```text
SetField rValue, rReceiver, @member8
```

Use the currently unimplemented `SetF` opcode position for `SetField` and
change it to the `ABC` operand format.

The VM handler resolves the member in the same way as `LoadField`, writes
`rValue` into the resolved receiver slot, and runs the GC write barrier for the
store.

#### `Invoke`

```text
Invoke rDst, rCallBase, @member8
```

`rCallBase` names a compiler-prepared contiguous call region:

```text
rDst       selected closure, then call result
rDst + 1   return-address slot
rCallBase  receiver
rCallBase + 1  first explicit argument
rCallBase + 2  second explicit argument
...
```

Code generation establishes:

```text
rCallBase = rDst + FRAME_HEADER_SIZE
```

With the existing two-word frame header, advancing with
`next_bp(bp, rDst)` maps the prepared region directly into the callee:

```text
callee r0 = receiver
callee r1 = first explicit argument
callee r2 = second explicit argument
...
```

The VM handler:

1. Reads the member-name string from the caller's constant table.
2. Loads the receiver from `rCallBase`.
3. Resolves the member slot through the receiver's runtime type value and image
   description.
4. Loads the ordinary method closure from that slot into `rDst`.
5. Keeps the selected closure, receiver, and arguments rooted in caller
   registers through the normal call-site GC poll.
6. Stores the return address in `rDst + 1`.
7. Advances the base pointer with `next_bp(bp, rDst)`.
8. Uses the selected closure's operations and constant table exactly as the
   existing ordinary application path does.

The existing return path writes the method result back to `rDst`. Invoke uses
the prepared register layout to establish the callee frame.

Place `Invoke` next to `Apply`, and place `LoadField` and `SetField` next to
`LoadFree` in the bytecode list.

### Member-call lowering

For `receiver.method(args)`:

1. Allocate `rDst`, the return-address slot, the receiver slot, and one slot per
   explicit argument as one contiguous call region.
2. Evaluate the receiver exactly once into `rDst + FRAME_HEADER_SIZE`.
3. Evaluate explicit arguments in source order into the following registers.
4. Emit:

```text
Invoke rDst, rDst + FRAME_HEADER_SIZE, @member8
```

A direct sibling-method call follows the same lowering, using the current
method's `self` as the receiver expression.

### GC treatment

The bytecode image owns the immutable type-description region and keeps its
member-name strings valid for the image's lifetime. A loaded description handle
is represented so the stack scanner recognizes it as image metadata.

The runtime type-value scanner shades:

- Every ordinary method closure stored in the type value.

The struct-instance scanner shades:

- Its runtime type-value pointer.
- Every pointer value in its member array.

`SetField` uses the collector's write barrier after updating an instance slot.
The selected closure written into `rDst` and the prepared receiver/arguments
remain ordinary stack roots while `Invoke` performs its call-site GC poll.
After the frame transition, `frame_rv` contains the selected ordinary closure,
so ordinary closure scanning and constant-table restoration continue to use the
existing frame representation.

### Bytecode and VM integration

Update the following together:

- Rust bytecode definitions, constructors, formatting, and operand visitors.
- The bytecode image's type-description region and its dumping and loading.
- Bytecode dumping, loading, and expect output.
- The C opcode list and dispatch table.
- VM handlers for `LoadType`, `LoadFree`, `LoadField`, `SetField`, and `Invoke`.
- Runtime type-value and struct-instance allocation, layout, member lookup, and
  GC scanning.
- Rust-to-C conversion of bytecode and the shared type-description region.
- The generated `vm.s` artifact.

The compiler maintains the invariant that each member operand selects a string
constant and each `Invoke` call area begins at `rDst + FRAME_HEADER_SIZE`.

### Verification

The current test runs `--inspect`, so it proves parsing and lowering. Add a
corresponding execution test whose result is `2029`, plus:

- Parser tests for member chaining and mixed initializers.
- Constructor tests for labeled, positional, reordered, duplicate, missing,
  unknown, and excess fields.
- Code-generation tests showing `self` in method `r0` and explicit arguments
  beginning at `r1`.
- Bytecode tests for `LoadType` and the three 8-bit member operations.
- A compile-error test for exhausting the 8-bit member constant index.
- A test showing that runtime type values share their image description while
  retaining their own ordinary method closures.
- A test showing that instances created from one runtime type value reuse its
  ordinary method closures.
- Repeated `Invoke` calls with stable allocation counts.
- Direct mutual recursion between two methods through `MemberRef`.
- Nested calls whose caller and callee methods use different constant tables.
- GC tests covering method closures held by runtime type values, instances
  holding member values, and pointer stores through `SetField`.
- Dump/load round trips containing the type-description region, `LoadType`,
  `LoadField`, `SetField`, and `Invoke`.

Targeted commands:

```text
cargo test
just expect tests/2_advanced_patterns/methods.qxq
cargo run -- tests/6_execution/methods.qxq
make -C vm rel
```
