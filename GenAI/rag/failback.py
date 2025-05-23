from typing import Any, Callable

from logs import logger

GenericFunction = Callable[..., Any]


def retry(attempts: int = 3) -> Callable[[GenericFunction], GenericFunction]:
    def decorator(fun: GenericFunction) -> GenericFunction:
        fun_name = fun.__name__

        def inner(*args, **kwargs):
            for i in range(attempts):
                try:
                    return fun(*args, **kwargs)
                except Exception as e:
                    state = f"{i + 1}/{attempts}"
                    logger.error(f"Failed to execute {fun_name} ({state}) - '{e}'")
            raise Exception(f"Function {fun_name} failed after {attempts} attempts")

        return inner

    return decorator
