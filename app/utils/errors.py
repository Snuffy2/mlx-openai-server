from http import HTTPStatus
from typing import Union


def create_error_response(
    message: str,
    err_type: str = "internal_error",
    status_code: Union[int, HTTPStatus] = HTTPStatus.INTERNAL_SERVER_ERROR,
    param: str = None,
    code: str = None,
):
    return {
        "error": {
            "message": message,
            "type": err_type,
            "param": param,
            "code": str(
                code
                or (
                    status_code.value
                    if isinstance(status_code, HTTPStatus)
                    else status_code
                )
            ),
        }
    }
