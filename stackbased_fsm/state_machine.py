# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import (
    Any,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
)


class StackContext:
    pass


TStackContext = TypeVar("TStackContext", bound=StackContext)


class StackedState(Generic[TStackContext]):
    __slots__ = ["_sm"]

    def __init__(self, sm: StateMachine[TStackContext]):
        self._sm = sm

    def __repr__(self) -> str:
        return self.__class__.__name__

    @property
    def sm(self) -> StateMachine[TStackContext]:
        return self._sm

    @property
    def context(self) -> TStackContext:
        return self._sm.context

    def enter(self) -> None:
        pass

    def exit(self) -> None:
        pass

    def reenter(self) -> None:
        pass

    def push(
        self,
        pushed: Union[
            List[Type[StackedState[TStackContext]]], Type[StackedState[TStackContext]]
        ],
    ) -> None:
        self.sm.push(pushed)

    def pop(self) -> None:
        try:
            self.sm.pop()
        except IndexError:
            pass

    def poppush(self, state_cls: Type[StackedState[TStackContext]]) -> None:
        self.sm.poppush(state_cls)


TStackedState = TypeVar("TStackedState", bound=StackedState[Any])
SStackedState = TypeVar("SStackedState", bound=StackedState[Any])


class GenericStackedState(
    StackedState[TStackContext], Generic[TStackContext, TStackedState]
):
    def __init__(self, sm: StateMachine[TStackContext], type_var: Type[TStackedState]):
        super().__init__(sm=sm)
        self.type_var = type_var


class StateMachineError(Exception):
    pass


class SequenceState(StackedState[TStackContext]):
    def __init__(
        self,
        sm: StateMachine[TStackContext],
        state_classes: List[Type[StackedState[TStackContext]]],
    ):
        super().__init__(sm=sm)
        self.state_classes = state_classes
        self.state_classes_iter = iter(state_classes)

    def enter(self) -> None:
        self._push_next()

    def reenter(self) -> None:
        self._push_next()

    def _push_next(self) -> None:
        try:
            self.push(next(self.state_classes_iter))
        except StopIteration:
            pass


class StateMachine(Generic[TStackContext]):
    def __init__(self, context: TStackContext):
        self._loop_count = 0
        self._context = context
        self.stack: List[StackedState[TStackContext]] = []
        self.just_pushed: Optional[StackedState[TStackContext]] = None
        self.just_popped: Optional[StackedState[TStackContext]] = None
        self.last_pushed: Optional[StackedState[TStackContext]] = None
        self.last_popped: Optional[StackedState[TStackContext]] = None

    @property
    def context(self) -> TStackContext:
        return self._context

    @property
    def loop_count(self) -> int:
        return self._loop_count

    def push(
        self,
        pushed: Union[
            List[Type[StackedState[TStackContext]]], Type[StackedState[TStackContext]]
        ],
    ) -> None:
        self.check_not_pushed_or_popped_twice()
        new_state = self._new_state_from_pushed_type(pushed)
        self.stack.append(new_state)
        self.just_pushed = new_state

    def _new_state_from_pushed_type(
        self,
        pushed: Union[
            List[Type[StackedState[TStackContext]]], Type[StackedState[TStackContext]]
        ],
    ) -> StackedState[TStackContext]:
        new_state: StackedState[TStackContext]
        if isinstance(pushed, list):
            new_state = SequenceState(sm=self, state_classes=pushed)
        elif isinstance(pushed, type):
            assert issubclass(pushed, StackedState), pushed
            new_state = pushed(sm=self)
        else:
            typing_origin: Type[
                GenericStackedState[TStackContext, StackedState[TStackContext]]
            ] = get_origin(pushed)
            assert issubclass(typing_origin, GenericStackedState)
            typing_args: Tuple[Type[StackedState[TStackContext]], ...] = get_args(
                pushed
            )
            assert len(typing_args) == 1
            type_var = typing_args[0]
            # assert issubclass(typing_args[0], StackedState)
            new_state = typing_origin(sm=self, type_var=type_var)
        return new_state

    def check_not_pushed_or_popped_twice(self) -> None:
        if self.just_popped or self.just_pushed:
            raise StateMachineError("Can't push or pop twice")

    def pop(self) -> None:
        self.check_not_pushed_or_popped_twice()
        self.just_popped = self.stack.pop()

    def poppush(self, pushed: Type[StackedState[TStackContext]]) -> None:
        self.check_not_pushed_or_popped_twice()
        self.just_popped = self.stack.pop()
        new_state = self._new_state_from_pushed_type(pushed)
        self.stack.append(new_state)
        self.just_pushed = new_state

    def run(self, state_cls: Type[StackedState[TStackContext]]) -> None:
        self.push(state_cls)
        while self.stack or self.just_popped:
            if self.just_pushed is None and self.just_popped is None:
                self.stack[-1].reenter()
            if (
                self.just_pushed is None
                and self.just_popped is None
                and self.just_pushed is None
            ):
                self.pop()
            just_popped, just_pushed = self.just_popped, self.just_pushed
            self.just_popped, self.just_pushed = None, None
            if just_popped is not None:
                self.last_popped = just_popped
                assert isinstance(just_popped, StackedState)
                just_popped.exit()
            if just_pushed is not None:
                self.last_pushed = just_pushed
                assert isinstance(just_pushed, StackedState), just_pushed
                just_pushed.enter()
            self._loop_count += 1

    def __repr__(self) -> str:
        state_repr_strs = [
            "State of state machine:",
            " - stack:",
        ]
        for i, state in reversed(list(enumerate(self.stack))):
            state_repr_strs.append(f"    [{i}] {state}")
        state_repr_strs.extend(
            [
                f" - last pushed: {self.last_pushed}",
                f" - last popped: {self.last_popped}",
            ]
        )
        return "\n".join(state_repr_strs)
