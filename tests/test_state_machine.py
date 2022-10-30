# -*- coding: utf-8 -*-
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

import pytest

from stackbased_fsm.state_machine import (
    GenericStackedState,
    StackContext,
    StackedState,
    StateMachine,
    StateMachineError,
)


class ExampleStackContext(StackContext):
    def __init__(self) -> None:
        self.a = ""
        self.b = ""
        self.c = 0  # for example 9
        self.d = 0  # for example 10
        self.e = 0  # for example 15


class ExampleState(StackedState[ExampleStackContext]):
    pass


class ExampleState1(ExampleState):
    def enter(self) -> None:
        self.context.a = "example 1 entered last"
        self.pop()

    def exit(self) -> None:
        self.context.b = "example 1 exited last"


class ExampleState2(ExampleState):
    # The state enters and exits without popping or pushing.
    def enter(self) -> None:
        self.context.a = "example 2 entered last"

    def exit(self) -> None:
        self.context.b = "example 2 exited last"


class ExampleState3(ExampleState):
    def enter(self) -> None:
        self.context.a = "example 3 entered last"
        self.poppush(ExampleState2)

    def exit(self) -> None:
        self.context.b = "example 3 exited last"


class ExampleState4(ExampleState):
    def enter(self) -> None:
        self.context.a = "example 4 entered last"
        self.push(ExampleState3)

    def exit(self) -> None:
        self.context.b = "example 4 exited last"


class ExampleState5(ExampleState):
    def enter(self) -> None:
        self.context.a = "example 5 entered last"
        self.push(ExampleState4)

    def exit(self) -> None:
        self.context.b = "example 5 exited last"


class ExampleState6(ExampleState):
    def enter(self) -> None:
        self.context.a = "example 6 entered last"

    def exit(self) -> None:
        self.context.b = "example 6 exited last"
        self.push(ExampleState2)


class ExampleState7(ExampleState):
    def enter(self) -> None:
        self.context.a = "example 7 entered last"

    def exit(self) -> None:
        self.context.b = "example 7 exited last"
        self.pop()


class ExampleState8(ExampleState):
    def enter(self) -> None:
        self.context.a = "example 8 entered last"
        self.push(ExampleState7)

    def reenter(self) -> None:
        raise Exception("Shouldn't get here")

    def exit(self) -> None:
        self.context.b = "example 8 exited last"


class ExampleState9(ExampleState):
    def enter(self) -> None:
        self.context.a = "example 9 entered last"
        self.push(ExampleState1)

    def reenter(self) -> None:
        if self.context.c < 6:
            self.context.c += 1
            self.push(ExampleState1)

    def exit(self) -> None:
        self.context.b = "example 9 exited last"


class ExampleState10(ExampleState):
    def enter(self) -> None:
        self.context.a = "example 10 entered last"
        self.push(ExampleState9)

    def reenter(self) -> None:
        if self.context.d < 1:
            self.context.c = 0
            self.context.d += 1
            self.poppush(ExampleState10)

    def exit(self) -> None:
        self.context.b = "example 10 exited last"


class ExampleState11(ExampleState):
    def enter(self) -> None:
        self.context.a = "example 11 entered last"
        self.push(
            [
                ExampleState1,
                ExampleState2,
                ExampleState3,
                ExampleState4,
                ExampleState5,
                ExampleState6,
                ExampleState9,
                ExampleState10,
            ]
        )

    def exit(self) -> None:
        self.context.b = "example 11 exited last"


class ExampleState12(ExampleState):
    def enter(self) -> None:
        self.context.a = "example 12 entered last"
        self.push(
            [
                ExampleState1,
                ExampleState7,
                ExampleState2,
                ExampleState3,
                ExampleState4,
                ExampleState5,
                ExampleState6,
                ExampleState9,
                ExampleState10,
            ]
        )

    def exit(self) -> None:
        self.context.b = "example 12 exited last"


TExampleState = TypeVar("TExampleState", bound=ExampleState)


class GenericExampleState(
    ExampleState, GenericStackedState[ExampleStackContext, TExampleState]
):
    pass


class ExampleState13(GenericExampleState[TExampleState]):
    def enter(self) -> None:
        self.context.a = "example 13 entered last"
        self.push(self.type_var)

    def exit(self) -> None:
        self.context.b = "example 13 exited last"


class ExampleState14(GenericExampleState[TExampleState]):
    def enter(self) -> None:
        self.context.a = "example 14 entered last"
        self.poppush(self.type_var)

    def exit(self) -> None:
        self.context.b = "example 14 exited last"


class ExampleWhileLoopState(GenericExampleState[TExampleState], ABC):
    def enter(self) -> None:
        self._enter_or_reenter()

    def reenter(self) -> None:
        self._enter_or_reenter()

    def _enter_or_reenter(self) -> None:
        if self.condition():
            self.step()
            self.push(self.type_var)

    @abstractmethod
    def condition(self) -> bool:
        pass

    def step(self) -> None:
        pass


class ExampleState15(ExampleWhileLoopState[TExampleState]):
    def condition(self) -> bool:
        return self.context.e < 10

    def step(self) -> None:
        self.context.e += 1

    def enter(self) -> None:
        self.context.a = "example 15 entered last"

    def exit(self) -> None:
        self.context.b = "example 15 exited last"


class ExampleState16(ExampleWhileLoopState[TExampleState]):
    def condition(self) -> bool:
        return True

    def enter(self) -> None:
        self.context.a = "example 16 entered last"

    def exit(self) -> None:
        self.context.b = "example 16 exited last"


class TestStateMachine:
    def setup_method(self) -> None:
        self.context = ExampleStackContext()
        self.sm = StateMachine[ExampleStackContext](self.context)
        assert self.sm.stack == []
        assert self.sm.just_pushed is None
        assert self.sm.just_popped is None
        assert self.sm.context.a == ""
        assert self.sm.context.b == ""

    def test_push(self) -> None:
        self.sm.push(ExampleState1)

        assert len(self.sm.stack) == 1
        assert isinstance(self.sm.stack[0], ExampleState1)
        assert isinstance(self.sm.just_pushed, ExampleState1)
        assert self.sm.just_pushed is self.sm.stack[0]
        assert self.sm.just_pushed.sm is self.sm
        assert self.sm.just_popped is None

        # Check can't push twice.
        with pytest.raises(StateMachineError):
            self.sm.push(ExampleState2)

        self.sm.just_pushed = None
        self.sm.just_popped = None

        self.sm.push(ExampleState2)
        assert len(self.sm.stack) == 2
        assert isinstance(self.sm.stack[0], ExampleState1)
        assert isinstance(self.sm.stack[1], ExampleState2)

    def test_pop(self) -> None:
        # Push, reset, push.
        self.sm.push(ExampleState1)
        self.sm.just_pushed = None
        self.sm.just_popped = None
        self.sm.push(ExampleState2)
        self.sm.just_pushed = None
        self.sm.just_popped = None

        assert len(self.sm.stack) == 2
        self.sm.pop()
        assert len(self.sm.stack) == 1

        assert isinstance(self.sm.just_popped, ExampleState2)
        assert self.sm.just_pushed is None
        assert len(self.sm.stack) == 1
        assert isinstance(self.sm.stack[0], ExampleState1)

        # Check can't pop twice.
        with pytest.raises(StateMachineError):
            self.sm.pop()

        self.sm.just_pushed = None
        self.sm.just_popped = None

        self.sm.pop()
        assert len(self.sm.stack) == 0

    def test_poppush(self) -> None:
        self.sm.push(ExampleState1)

        self.sm.just_pushed = None
        self.sm.just_popped = None

        self.sm.poppush(ExampleState2)

        assert len(self.sm.stack) == 1
        assert isinstance(self.sm.stack[0], ExampleState2)
        assert isinstance(self.sm.just_popped, ExampleState1)
        assert isinstance(self.sm.just_pushed, ExampleState2)

        # Check can't pop push twice.
        with pytest.raises(StateMachineError):
            self.sm.poppush(ExampleState2)

    def test_run_example1(self) -> None:
        # Example 1 enters, then pops itself, so then exits.
        self.sm.run(ExampleState1)
        assert self.context.a == "example 1 entered last"
        assert self.context.b == "example 1 exited last"
        assert self.sm.loop_count == 2

    def test_run_example2(self) -> None:
        # Example 2 enters, is popped automatically, so then exits.
        self.sm.run(ExampleState2)
        assert self.context.a == "example 2 entered last"
        assert self.context.b == "example 2 exited last"
        assert self.sm.loop_count == 2

    def test_run_example3(self) -> None:
        # Example 3 enters, then pops itself and pushes Example 2.
        self.sm.run(ExampleState3)
        assert self.context.a == "example 2 entered last"
        assert self.context.b == "example 2 exited last"
        assert self.sm.loop_count == 3

    def test_run_example4(self) -> None:
        # Example 4 enters, then pushes Example 3, then is popped automatically.
        self.sm.run(ExampleState4)
        assert self.context.a == "example 2 entered last"
        assert self.context.b == "example 4 exited last"
        assert self.sm.loop_count == 5

    def test_run_example5(self) -> None:
        # Example 5 enters, then pushes Example 4, then is popped automatically.
        self.sm.run(ExampleState5)
        assert self.context.a == "example 2 entered last"
        assert self.context.b == "example 5 exited last"
        assert self.sm.loop_count == 7

    def test_run_example6(self) -> None:
        # Example 6 pushes Example 2 on exit, so it exits last.
        self.sm.run(ExampleState6)
        assert self.context.a == "example 2 entered last"
        assert self.context.b == "example 2 exited last"
        assert self.sm.loop_count == 4

    def test_run_example7(self) -> None:
        # Example 7 pops on exit.
        self.sm.run(ExampleState7)
        assert self.context.a == "example 7 entered last"
        assert self.context.b == "example 7 exited last"
        assert self.sm.loop_count == 2

    def test_run_example8(self) -> None:
        # Example 8 pushes Example 7, so it gets popped before it can reenter.
        self.sm.run(ExampleState8)
        assert self.context.a == "example 7 entered last"
        assert self.context.b == "example 8 exited last"
        assert self.sm.loop_count == 4

    def test_run_example9(self) -> None:
        # Example 9 pushes Example 1 six times, then pops itself.
        self.sm.run(ExampleState9)
        assert self.context.a == "example 1 entered last"
        assert self.context.b == "example 9 exited last"
        assert self.context.c == 6
        assert self.sm.loop_count == 16

    def test_run_example10(self) -> None:
        # Example 10 pushes Example 9, then poppushes itself once.
        self.sm.run(ExampleState10)
        assert self.context.a == "example 1 entered last"
        assert self.context.b == "example 10 exited last"
        assert self.context.c == 6
        assert self.context.d == 1
        assert self.sm.loop_count == 35

    def test_run_example11(self) -> None:
        # Example 11 pushes a list of states.
        self.sm.run(ExampleState11)
        assert self.context.a == "example 1 entered last"
        assert self.context.b == "example 11 exited last"
        assert self.sm.loop_count == 66

    def test_run_example12(self) -> None:
        # Example 12 pushes a list including example 7, so it gets popped early.
        self.sm.run(ExampleState12)
        assert self.context.a == "example 7 entered last"
        assert self.context.b == "example 12 exited last"
        assert self.sm.loop_count == 8

    def test_run_example13(self) -> None:
        # Example 13 pushes its type var, so it exits last.
        self.sm.run(ExampleState13[ExampleState1])
        assert self.context.a == "example 1 entered last"
        assert self.context.b == "example 13 exited last"
        assert self.sm.loop_count == 4

        self.sm.run(ExampleState13[ExampleState13[ExampleState2]])
        assert self.context.a == "example 2 entered last"
        assert self.context.b == "example 13 exited last"
        assert self.sm.loop_count == 10

    def test_run_example14(self) -> None:
        # Example 14 poppushes its type var so its type var exits last.
        self.sm.run(ExampleState14[ExampleState1])
        assert self.context.a == "example 1 entered last"
        assert self.context.b == "example 1 exited last"
        assert self.sm.loop_count == 3

        self.sm.run(ExampleState14[ExampleState13[ExampleState2]])
        assert self.context.a == "example 2 entered last"
        assert self.context.b == "example 13 exited last"
        assert self.sm.loop_count == 8

    def test_run_example15(self) -> None:
        # Example 15 loops its type var until e = 10.
        self.sm.run(ExampleState15[ExampleState1])
        assert self.context.a == "example 1 entered last"
        assert self.context.b == "example 15 exited last"
        assert self.sm.context.e == 10
        assert self.sm.loop_count == 22

    def test_run_example16(self) -> None:
        # Example 15 loops its type var until it gets popped.
        self.sm.run(ExampleState16[ExampleState7])
        assert self.context.a == "example 7 entered last"
        assert self.context.b == "example 16 exited last"
        assert self.sm.loop_count == 4

        # These don't work because the sequence doesn't pop on exit.
        # self.sm.run(ExampleState16[ExampleState11])
        # # assert self.context.a == "example 7 entered last"
        # # assert self.context.b == "example 16 exited last"
        # assert self.sm.loop_count == 4
        #
        # self.sm.run(ExampleState16[ExampleState12])
        # # assert self.context.a == "example 7 entered last"
        # # assert self.context.b == "example 16 exited last"
        # assert self.sm.loop_count == 4