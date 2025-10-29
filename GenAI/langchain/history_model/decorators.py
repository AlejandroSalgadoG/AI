from tenacity import retry, stop_after_attempt, RetryCallState
from typing import Callable


def log_and_none(msg: str) -> Callable:
    def wrapper(retry_state: RetryCallState) -> None:
        print(msg)
        if retry_state.next_action is None:
            raise retry_state.outcome.exception()
        return None
    return wrapper


def retry_and_log(attepts: int, msg_on_error: str, msg_on_failure: str, **kwargs) -> Callable:
    return retry(
        stop=stop_after_attempt(attepts),
        before_sleep=log_and_none(msg_on_error),
        retry_error_callback=log_and_none(msg_on_failure),
        **kwargs,
    )
