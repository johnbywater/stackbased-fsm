# -*- coding: utf-8 -*-
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    Any,
    Generic,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from typing_extensions import TypeVarTuple, Unpack

DType = TypeVar("DType")
Shape = TypeVarTuple("Shape")

# class Array(Generic[DType, *Shape]):


# class Steps(_TupleType(tuple, -1, inst=False, name="Tuple"), Generic[TypeVars]):
#     pass


class Context:
    pass


TContext = TypeVar("TContext", bound=Context, covariant=True)


class State(Generic[TContext], ABC):
    __slots__ = ["_sm"]

    def __init__(self, sm: StateMachine[TContext]):
        self._sm = sm

    def __repr__(self) -> str:
        return self.__class__.__name__

    @property
    def sm(self) -> StateMachine[TContext]:
        return self._sm

    @property
    def context(self) -> TContext:
        return self._sm.context

    @abstractmethod
    def enter(self) -> None:
        pass  # pragma: no cover

    def suspend(self):
        pass

    def resume(self) -> None:
        pass

    def exit(self) -> None:
        pass

    def push(self, pushed: PushableTypes) -> None:
        self.sm.push(pushed)

    def pop(self) -> None:
        try:
            self.sm.pop()
        except IndexError:
            pass

    def poppush(self, pushed: PushableTypes) -> None:
        self.sm.poppush(pushed)

    @classmethod
    def construct_from_type(cls, sm: StateMachine) -> State:
        return cls(sm=sm)

    @classmethod
    def construct_from_generic_alias(cls, sm: StateMachine, alias: object) -> State:
        return cls(sm=sm)


StateAlias = State[Context]

TStates = TypeVarTuple("TStates")


class Steps(Generic[Unpack[TStates]]):
    pass


TState = TypeVar(
    "TState",
    bound=Union[
        StateAlias, Steps, Tuple[Union[StateAlias, Tuple[StateAlias, ...]], ...]
    ],
)


PushableTypes = Union[
    Type[StateAlias],
    Sequence[Type[StateAlias]],
    Type[Sequence[StateAlias]],
    Type[Steps],
]


class Block(State[TContext], Generic[TContext, TState]):
    def __init__(
        self,
        sm: StateMachine[TContext],
        tv_block: PushableTypes,
    ):
        super().__init__(sm=sm)
        self._tv_block = tv_block

    @abstractmethod
    def enter(self) -> None:
        pass  # pragma: no cover

    @classmethod
    def construct_from_type(cls, sm: StateMachine) -> Block:
        # Todo: Not sure if this is good enough, but works for now...
        # Look for type variable in original bases.
        orig_bases__ = cls.__orig_bases__  # type: ignore
        for base_type in orig_bases__:
            type_args = get_args(base_type)
            if len(type_args) == 1 and not isinstance(type_args[0], TypeVar):
                return cls(sm=sm, tv_block=type_args[0])
        else:
            raise TypeError(f"pushed type '{cls}' is an unsubscripted {Block}")

    @classmethod
    def construct_from_generic_alias(cls, sm: StateMachine, alias: object) -> Block:
        typing_args: Tuple[Type[State[TContext]], ...] = get_args(alias)
        if not isinstance(typing_args[-1], TypeVar):
            tv_block = typing_args[-1]
            # try:
            #     assert issubclass(get_origin(tv_block), (State, tuple)), tv_block
            # except AssertionError:
            #     raise

            new_state = cls(sm=sm, tv_block=tv_block)
            return new_state
        else:
            raise TypeError(
                f"parameter of {alias} is not a pushable type: {typing_args[-1]}"
            )


class _SequenceOfStates(State[TContext]):
    """
    Constructed by a Tuple type construct or a Sequence object.
    """

    def __init__(
        self,
        sm: StateMachine[TContext],
        tv_sequence: Sequence[Type[State[TContext]]],
    ):
        super().__init__(sm=sm)
        self._tv_sequence = tv_sequence
        self._tv_iter = iter(tv_sequence)

    def enter(self) -> None:
        try:
            self.push(next(self._tv_iter))
        except StopIteration:
            pass

    def resume(self) -> None:
        self.enter()


class StateMachine(Generic[TContext]):
    def __init__(self, context: TContext):
        self._context = context
        self._loop_count = 0
        self.stack: List[StateAlias] = []
        self.just_pushed: Optional[StateAlias] = None
        self.just_popped: Optional[StateAlias] = None
        self.last_pushed: Optional[StateAlias] = None
        self.last_popped: Optional[StateAlias] = None

    @property
    def context(self) -> TContext:
        return self._context

    @property
    def loop_count(self) -> int:
        return self._loop_count

    def push(self, pushed: PushableTypes) -> None:
        assert pushed is not None
        self._check_not_pushed_or_popped_twice()
        new_state = self._new_state_from_pushed_type(pushed)
        self.stack.append(new_state)
        self.just_pushed = new_state

    def pop(self) -> None:
        self._check_not_pushed_or_popped_twice()
        self.just_popped = self.stack.pop()

    def poppush(self, pushed: PushableTypes) -> None:
        assert pushed is not None
        self._check_not_pushed_or_popped_twice()
        self.just_popped = self.stack.pop()
        new_state = self._new_state_from_pushed_type(pushed)
        self.stack.append(new_state)
        self.just_pushed = new_state

    def run(
        self,
        initial: PushableTypes,
    ) -> None:
        self.push(initial)
        while self.stack or self.just_popped:
            if self.just_pushed is None and self.just_popped is None:
                print("Resumed:", self.stack[-1])
                self.stack[-1].resume()
            if (
                self.just_pushed is None
                and self.just_popped is None
                and self.just_pushed is None
            ):
                self.pop()
            just_popped, just_pushed = self.just_popped, self.just_pushed
            self.just_popped, self.just_pushed = None, None
            if just_popped is not None:
                # print("Popped:", just_popped)
                self.last_popped = just_popped
                assert isinstance(just_popped, State)
                print("Exited:", just_popped)
                just_popped.exit()
            if just_pushed is not None:
                if just_popped is None:
                    if len(self.stack) > 1:
                        print("Suspended:", self.stack[-2])
                        self.stack[-2].suspend()
                # print("Pushed:", just_pushed)
                self.last_pushed = just_pushed
                assert isinstance(just_pushed, State), just_pushed
                print("Entered:", just_pushed)
                just_pushed.enter()
            self._loop_count += 1

    def _check_not_pushed_or_popped_twice(self) -> None:
        if self.just_popped or self.just_pushed:
            raise StateMachineError("Can't push or pop twice")

    def _new_state_from_pushed_type(self, pushed: PushableTypes) -> StateAlias:
        new_state: StateAlias
        if isinstance(pushed, Sequence):
            return _SequenceOfStates(sm=self, tv_sequence=pushed)
        elif isinstance(pushed, type):
            if not issubclass(pushed, State):
                raise TypeError(
                    f"pushed type '{pushed.__name__}' not a subclass of {State}"
                )
            else:
                return pushed.construct_from_type(sm=self)
        else:
            # We got an object, hopefully a subscripted type.
            typing_origin = get_origin(pushed)
            typing_args: Tuple[Type[State[TContext]], ...] = get_args(pushed)
            if isinstance(typing_origin, type):
                if issubclass(typing_origin, tuple):
                    if Ellipsis in typing_args:
                        raise TypeError(
                            f"pushed '{pushed}' has Ellipsis and so not a variadic"
                            " Tuple"
                        )

                    return _SequenceOfStates(sm=self, tv_sequence=typing_args)
                elif issubclass(typing_origin, Steps):
                    return _SequenceOfStates(sm=self, tv_sequence=typing_args)

                elif issubclass(typing_origin, State):
                    return typing_origin.construct_from_generic_alias(
                        sm=self, alias=pushed
                    )
            # if typing_origin is None:
            raise TypeError(f"pushed object '{pushed}' not supported")
            # raise TypeError(f"generic alias '{pushed}' not supported")

    def __str__(self) -> str:
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


class StateMachineError(Exception):
    pass


class ConditionState(State[TContext]):
    def enter(self) -> None:
        conditioned_state = self.sm.stack[-2]
        assert isinstance(conditioned_state, _ConditionedBlock), conditioned_state
        conditioned_state.set_condition_result(self.condition())
        self.pop()

    @abstractmethod
    def condition(self) -> bool:
        pass  # pragma: no cover


TConditionState = TypeVar(
    "TConditionState", bound=ConditionState[Context], covariant=True
)


class _ConditionedBlock(Block[TContext, TState]):
    def __init__(
        self,
        sm: StateMachine[TContext],
        tv_block: PushableTypes,
    ):
        super().__init__(sm=sm, tv_block=tv_block)

    @abstractmethod
    def set_condition_result(self, value: bool) -> None:
        pass  # pragma: no cover


class _ConditionalBlock(
    _ConditionedBlock[TContext, TState], Generic[TContext, TConditionState, TState]
):
    CONDITION_RESULT_INITIAL_VALUE: bool

    def __init__(
        self,
        sm: StateMachine[TContext],
        tv_block: PushableTypes,
        tv_condition: Type[ConditionState[TContext]],
    ):
        super().__init__(sm=sm, tv_block=tv_block)
        self._condition_result: bool = self.CONDITION_RESULT_INITIAL_VALUE
        self._tv_condition = tv_condition
        self._just_pushed_condition = False

    def set_condition_result(self, value: bool) -> None:
        self._condition_result = value

    def _has_condition_been_met(self) -> bool:
        return self._condition_result is not self.CONDITION_RESULT_INITIAL_VALUE

    @classmethod
    def construct_from_generic_alias(
        cls, sm: StateMachine, alias: object
    ) -> _ConditionalBlock:
        typing_args = get_args(alias)
        assert len(typing_args) == 2
        tv_condition = typing_args[0]
        # assert issubclass(tv_condition, ConditionState)
        tv_block = typing_args[1]
        # assert issubclass(tv_condition, State)
        return cls(sm=sm, tv_block=tv_block, tv_condition=tv_condition)


class _ConditionalLoop(_ConditionalBlock[TContext, TConditionState, TState]):
    def enter(self) -> None:
        if self._has_condition_been_met():
            self.pop()
        elif not self._just_pushed_condition:
            self.push(self._tv_condition)
            self._just_pushed_condition = True
        else:
            self.push(self._tv_block)
            self._just_pushed_condition = False

    def resume(self) -> None:
        self.enter()

    @classmethod
    def construct_from_type(cls, sm: StateMachine) -> _ConditionalLoop:
        # Look for type variable in original bases.
        # Todo: Not sure if this is good enough, but works for now...
        orig_bases__ = cls.__orig_bases__  # type: ignore
        assert len(orig_bases__) == 1
        base_type = orig_bases__[0]
        base_origin = get_origin(base_type)
        assert isinstance(base_origin, type) and issubclass(
            base_origin, _ConditionalLoop
        )
        type_args = get_args(base_type)
        assert len(type_args) == 2
        assert all([not isinstance(tv, TypeVar) for tv in type_args]), type_args

        tv_condition = type_args[0]
        # origin_tv_block = get_origin(tv_condition)
        # assert issubclass(origin_tv_block, ConditionState)

        tv_block = type_args[1]
        # origin_tv_block = get_origin(tv_block)
        # assert origin_tv_block is tuple
        return cls(sm=sm, tv_block=tv_block, tv_condition=tv_condition)


class RepeatUntil(_ConditionalLoop[Context, TConditionState, TState]):
    CONDITION_RESULT_INITIAL_VALUE = False


Until = RepeatUntil


class RepeatWhile(_ConditionalLoop[Context, TConditionState, TState]):
    CONDITION_RESULT_INITIAL_VALUE = True


While = RepeatWhile


class DoStepsUntil(_ConditionalBlock[Context, TConditionState, TState]):
    CONDITION_RESULT_INITIAL_VALUE = False

    def enter(self) -> None:
        args = get_args(self._tv_block)
        block_origin = get_origin(self._tv_block)
        if block_origin not in (tuple, Steps):
            raise TypeError(f"Type variable origin is not {tuple}: {self._tv_block}")
        assert args, self._tv_block
        self._iter = iter(args)
        self._next()

    def resume(self) -> None:
        self._next()

    def _next(self) -> None:
        if self._has_condition_been_met():
            self.pop()
        elif not self._just_pushed_condition:
            self.push(self._tv_condition)
            self._just_pushed_condition = True
        else:
            try:
                next_state = next(self._iter)
                self.push(next_state)
            except StopIteration:
                self.pop()
            self._just_pushed_condition = False


class AnyCondition(ConditionState[Context], _ConditionedBlock[Context, TState]):
    def __init__(
        self,
        sm: StateMachine[TContext],
        tv_block: PushableTypes,
    ):
        super().__init__(sm=sm, tv_block=tv_block)
        self._condition_result = False

    def set_condition_result(self, value: bool) -> None:
        self._condition_result = value

    def enter(self) -> None:
        conditioned_state = self.sm.stack[-2]
        assert isinstance(conditioned_state, _ConditionedBlock)
        self._iter = iter(get_args(self._tv_block))
        self._next()

    def resume(self) -> None:
        self._next()

    def _next(self) -> None:
        if self._condition_result:
            conditioned_state = self.sm.stack[-2]
            assert isinstance(conditioned_state, _ConditionedBlock)
            conditioned_state.set_condition_result(self.condition())
            self.pop()
        else:
            try:
                next_state = next(self._iter)
                # assert issubclass(next_state, ConditionState)
                self.push(next_state)
            except StopIteration:
                self.pop()

    def condition(self) -> bool:
        return self._condition_result


class AllConditions(ConditionState[Context], _ConditionedBlock[Context, TState]):
    def __init__(
        self,
        sm: StateMachine[TContext],
        tv_block: PushableTypes,
    ):
        super().__init__(sm=sm, tv_block=tv_block)
        self._condition_results: List[bool] = []

    def set_condition_result(self, value: bool) -> None:
        self._condition_results.append(value)

    def enter(self) -> None:
        conditioned_state = self.sm.stack[-2]
        assert isinstance(conditioned_state, _ConditionedBlock)
        self._iter = iter(get_args(self._tv_block))
        self._next()

    def resume(self) -> None:
        self._next()

    def _next(self) -> None:
        try:
            next_state = next(self._iter)
            # assert issubclass(next_state, ConditionState)
            self.push(next_state)
        except StopIteration:
            self.pop()

    def exit(self) -> None:
        conditioned_state = self.sm.stack[-1]
        assert isinstance(conditioned_state, _ConditionedBlock)
        conditioned_state.set_condition_result(self.condition())

    def condition(self) -> bool:
        return all(self._condition_results)


Or = AnyCondition

TLiteral = TypeVar("TLiteral")


class LiteralCondition(ConditionState[TContext], Generic[TContext, TLiteral]):
    def __init__(self, sm: StateMachine, literal: Any):
        super().__init__(sm=sm)
        self.literal = literal

    @classmethod
    def construct_from_type(cls, sm: StateMachine) -> State:
        orig_bases__ = cls.__orig_bases__  # type: ignore
        assert len(orig_bases__) == 1
        base_type = orig_bases__[0]
        type_args = get_args(base_type)
        assert len(type_args) == 1 and not isinstance(type_args[0], TypeVar)
        literal_type = type_args[0]
        literal_args = get_args(literal_type)
        return cls(sm, *literal_args)

    @classmethod
    def construct_from_generic_alias(
        cls, sm: StateMachine, alias: object
    ) -> LiteralCondition:
        literal_type = get_args(alias)[0]
        literal_args = get_args(literal_type)
        return cls(sm, *literal_args)

    @abstractmethod
    def condition(self) -> bool:
        pass  # pragma: no cover
