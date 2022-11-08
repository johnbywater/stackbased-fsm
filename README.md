# Welcome to Stack-Based Finite State Machine

This project provides a stack-based finite state machine.

A state machine has a stack. States are pushed onto and popped
from stack. The state machine will call 'enter' and 'exit' methods on the
states when they are 'pushed' and 'popped'. The states will then push and pop
other states, according to their individual implementation. States have
access to the stack, and also to a context object containing "global" variables.

A few basic state classes are also provided, with which "programs"
can be defined. Programs define a nested set of state types, from which
states will be constructed.


## Installation

    $ pip install stackbased-fsm

## Getting started

We can use the `Context` class to define an example context object.

```python
from stackbased_fsm import Context

class ExampleContext(Context):
    def __init__(self):
        self.a = 1
        self.b = "2"
        self.c = []


context = ExampleContext()
```

We can use the `StateMachine` class to construct state machine object that uses the
context object.

```python
from stackbased_fsm import StateMachine

sm = StateMachine(context=context)
```

The state machine has a stack of states. All states on the state machine's stack will
have access to the context object. The attributes of the context object are like
the "global" variables for the states of the state machine.

We can use the `State` class to define types of state for our state machine. It is a
generic class, which has one type variable that is expected to be a context class.

```python
from stackbased_fsm import State

class ExampleState(State[ExampleContext]):
    pass
```

We can use the `ExampleState` as a base class to define `IncrementA`, `AssignB`, and
`AppendC` which will increment `a`, assign to `b`, and append to `c` respectively.

```python

class IncrementA(ExampleState):
    def enter(self) -> None:
        self.context.a += 1
        self.pop()


class AssignB(ExampleState):
    def enter(self) -> None:
        self.context.b = "def"
        self.pop()


class AppendC(ExampleState):
    def enter(self) -> None:
        self.context.c.append("xyz")
        self.pop()
```

The state machine has a `run()` method which can be used by a client to pass
a "program" to the state machine. A program is a type construct, in the simplest
case a single state class, and more usually a nested set of subscripted generic
state classes (see below).

```python
sm.run(IncrementA)

assert context.a == 2
```

We can use the `SequenceOfStates` class to define a sequence of states. It is a
variadic generic state class, and so can take any number of state classes as its
type arguments.

```python
from stackbased_fsm import SequenceOfStates

ExampleSequence = SequenceOfStates[IncrementA, AssignB, AppendC]
```

We can run the sequence and check the context has been updated.

```python
sm.run(ExampleSequence)

assert context.a == 3
assert context.b == "def"
assert context.c == ["xyz"]
```

The state machine object has methods to `push()`, `pop()`, and `poppush()`
states on the stack. State objects have four methods, `enter()`, `exit()`,
`suspend()` and `resume()`.

When a program is passed to the state machine using the `run()` method, a state object
is constructed from the root type of the program (a type of state). The state object
is then pushed onto the stack, and its `enter()` method is called. The state machine
then iterates over the stack, detecting when states have been pushed and popped,
calling methods on the stacked states accordingly, until the stack is empty.

After a state has been pushed onto the stack, the state's `enter()` method
will be called. After a state is popped off the stack, the state's `exit()` method will
be called. A state's `suspend()` method will be called when another state is pushed
on top of it, and its `resume()` method will be called after that state is popped off.

For example, the `SequenceOfStates` works in the following way. When it is pushed onto
the stack, its `enter()` method is called. Its `enter()` method will push the first
item in the sequence onto the stack. Its `suspend()` method is then called, and then
the `enter()` method of the first item is called. If the pushed state neither pushes
or pops another state, it will be automatically popped and its `exit()` method will
be called. When that state is popped, the `resume()` method of the sequence will be
called, which will push the next item in the sequence onto the stack. After all items
have been pushed onto the stack, the sequence's `resume()` method will call `pop()`,
which will result in itself being popped off the stack. This may result in an empty
stack, and the end of a program.


## Conditions and conditioned states

The "condition" state class `ConditionState` can be used to define conditions.
When the `enter()` method of a condition state is called, its `condition()` method
will be called. The `condition()` method is abstract on the `ConditionState` class
and is expected to be implemented on subclasses. This method is expected to return
a Boolean value (*true* or *false*). This value will be used by the condition state's `enter()`
method to call the `set_condition_result()` method of state below it on the stack,
which is expected to be a "conditioned" state, and therefore have such a method.

In the example below, the class `AIsLessThan5` has a `condition()` method that
returns `True` if `a` is less than 5.

```python
from stackbased_fsm import ConditionState


class AIsLessThan5(ConditionState):
    def condition(self):
        return self.context.a < 5
```

Conditions are used by conditioned states. For example, the classes
`RepeatUntil`, `RepeatWhile` are conditioned states. These conditioned
states take two type variables. The first type variable is expected to
be a type of condition. The second type variable is expected to be a type
of state. These conditioned states alternate between pushing the condition
state and then pushing the other state. `RepeatUntil` continues in this way
until the condition is *true*. `RepeatWhile` continues in this way until
the condition is *false*.

In the example below, `ExampleLoop` will push `IncrementA` again and again
so long as `a` is less than 5.

```python
from stackbased_fsm import RepeatWhile

ExampleLoop = RepeatWhile[AIsLessThan5, IncrementA]

sm.run(ExampleLoop)
```

The result is the value of `a` is 5.

```python
assert context.a == 5, context.a
```

Conditions can be grouped with `AnyCondition` and `AllConditions` (aliased as
`Or` and `And` respectively).


## Developers

### Install Poetry

The first thing is to check you have Poetry installed.

    $ poetry --version

If you don't, then please [install Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer).

It will help to make sure Poetry's bin directory is in your `PATH` environment variable.

But in any case, make sure you know the path to the `poetry` executable. The Poetry
installer tells you where it has been installed, and how to configure your shell.

Please refer to the [Poetry docs](https://python-poetry.org/docs/) for guidance on
using Poetry.

### Setup for PyCharm users

You can easily obtain the project files using PyCharm (menu "Git > Clone...").
PyCharm will then usually prompt you to open the project.

Open the project in a new window. PyCharm will then usually prompt you to create
a new virtual environment.

Create a new Poetry virtual environment for the project. If PyCharm doesn't already
know where your `poetry` executable is, then set the path to your `poetry` executable
in the "New Poetry Environment" form input field labelled "Poetry executable". In the
"New Poetry Environment" form, you will also have the opportunity to select which
Python executable will be used by the virtual environment.

PyCharm will then create a new Poetry virtual environment for your project, using
a particular version of Python, and also install into this virtual environment the
project's package dependencies according to the `pyproject.toml` file, or the
`poetry.lock` file if that exists in the project files.

You can add different Poetry environments for different Python versions, and switch
between them using the "Python Interpreter" settings of PyCharm. If you want to use
a version of Python that isn't installed, either use your favourite package manager,
or install Python by downloading an installer for recent versions of Python directly
from the [Python website](https://www.python.org/downloads/).

Once project dependencies have been installed, you should be able to run tests
from within PyCharm (right-click on the `tests` folder and select the 'Run' option).

Because of a conflict between pytest and PyCharm's debugger and the coverage tool,
you may need to add ``--no-cov`` as an option to the test runner template. Alternatively,
just use the Python Standard Library's ``unittest`` module.

You should also be able to open a terminal window in PyCharm, and run the project's
Makefile commands from the command line (see below).

### Setup from command line

Obtain the project files, using Git or suitable alternative.

In a terminal application, change your current working directory
to the root folder of the project files. There should be a Makefile
in this folder.

Use the Makefile to create a new Poetry virtual environment for the
project and install the project's package dependencies into it,
using the following command.

    $ make install-packages

It's also possible to also install the project in 'editable mode'.

    $ make install

Please note, if you create the virtual environment in this way, and then try to
open the project in PyCharm and configure the project to use this virtual
environment as an "Existing Poetry Environment", PyCharm sometimes has some
issues (don't know why) which might be problematic. If you encounter such
issues, you can resolve these issues by deleting the virtual environment
and creating the Poetry virtual environment using PyCharm (see above).

### Project Makefile commands

You can run tests using the following command.

    $ make test

You can check the formatting of the code using the following command.

    $ make lint

You can reformat the code using the following command.

    $ make fmt

Tests belong in `./tests`. Code-under-test belongs in `./stackbased_fsm`.

See the [Python eventsourcing project](https://github.com/pyeventsourcing/eventsourcing)
for more information and guidance about developing event-sourced applications.

Edit package dependencies in `pyproject.toml`. Update installed packages (and the
`poetry.lock` file) using the following command.

    $ make update-packages
