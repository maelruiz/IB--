# IB++ Programming Language Syntax Sheet

## Literals
- **Numbers:** `123`, `3.14`
- **Strings:** `"hello"`, `"abc\nxyz"`
- **Booleans:** `true`, `false`
- **Null:** `null`
- **Lists:** `[1, 2, 3]`, `["a", "b"]`
- **Collectoins:** `new Collection()`

## Variables
```pseudo
NAME = value
```
- Variable names are case-sensitive.

## Arithmetic Operators
- `+`, `-`, `*`, `/`, `div` (integer division), `mod` (modulo), `^` (power)

## Comparison Operators
- `==`, `!=`, `<`, `>`, `<=`, `>=`

## Logical Operators
- `and`, `or`, `not`

## Assignment
```pseudo
A = 5
```

## List Access and Assignment
```pseudo
L = [1,2,3]
output L[0] // 1
L[1] = 42
```

## If Statements
```pseudo
if CONDITION then
    STATEMENTS
elif CONDITION then
    STATEMENTS
else
    STATEMENTS
end if
```
- `elif` and `else` are optional.

## While Loops
```pseudo
loop while CONDITION
    STATEMENTS
end loop
```

## Loop (For-like)
```pseudo
loop I from START to END
    STATEMENTS
end loop
```
- `I` is the loop variable.

## Functions (Methods)
```pseudo
method NAME(ARG1, ARG2)
    STATEMENTS
    return VALUE
end method
```
- **Auto-return:**  
  `method NAME(ARG1, ARG2) -> EXPR`

## Function Call
```pseudo
NAME(ARG1, ARG2)
```

## Return, Continue, Break
- `return VALUE`
- `continue`
- `break`

## Input/Output
```pseudo
input VAR
output EXPR1, EXPR2, ...
```

## Built-in Functions
- `print(value)` — prints value
- `print_ret(value)` — returns string of value
- `input(prompt)` — input from user
- `input_int(prompt)` — input, converted to int
- `clear()` / `cls()` — clear screen
- `is_number(value)` — true if number
- `is_string(value)` — true if string
- `is_list(value)` — true if list
- `is_function(value)` — true if function
- `append(list, value)` — append to list
- `pop(list)` — pop from list
- `extend(list, list2)` — extend list
- `len(list or string)` — length
- `.length()` — length
- `str(value)` — convert to string
- `int(value)` — convert to int
- `random(a, b)` — random int in [a, b]
- `run(filename)` — run another script

## String Methods
- `S.length()`
- `S.upper()`
- `S.lower()`
- `S.split([sep])`

## List Methods
- `L.length()`
- `L.append(value)`
- `L.pop()`
- `L.extend(list)`

## Collection Methods
- `COLL.length()`
- `COLL.getNext()`
- `COLL.hasNext()`
- `COLL.addItem(ITEM)`
- `COLL.resetNext()`

## Comments
```pseudo
// This is a comment
```

## Other Notes
- All statements end by newline, not semicolon.
- Indentation is not significant, but recommended for readability.
- All keywords are lowercase.
