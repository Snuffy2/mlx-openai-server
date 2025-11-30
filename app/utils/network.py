"""Network utility functions."""

from __future__ import annotations

from contextlib import suppress
import ipaddress
import socket

from ..const import DEFAULT_BIND_HOST


def is_port_available(host: str | None = None, port: int | None = None) -> bool:
    """Check if a port is available for binding on the specified host.

    Parameters
    ----------
    host : str, optional
        The host to check. If None, defaults to DEFAULT_BIND_HOST.
    port : int
        The port to check.

    Returns
    -------
    bool
        True if the port is available, False otherwise.

    Raises
    ------
    ValueError
        If port is None.
    """
    if port is None:
        raise ValueError("port must be specified")

    if host is None:
        host = DEFAULT_BIND_HOST

    family = socket.AF_INET6 if _is_ipv6_host(host) else socket.AF_INET
    bind_host = _normalize_host_for_binding(host, family)
    sock: socket.socket | None = None
    try:
        sock = socket.socket(family, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        bind_address = (bind_host, port, 0, 0) if family == socket.AF_INET6 else (bind_host, port)
        sock.bind(bind_address)
    except OSError:
        return False
    finally:
        if sock is not None:
            with suppress(Exception):
                sock.close()
    return True


def _is_ipv6_host(host: str) -> bool:
    """
    Determine whether the provided host string denotes an IPv6 address. Surrounding whitespace and enclosing brackets (`[...]`) are ignored.
    
    Parameters:
        host (str): Host string to inspect; may be an IPv6 literal with or without surrounding brackets.
    
    Returns:
        `True` if the host is an IPv6 address, `False` otherwise.
    """
    value = host.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    try:
        addr = ipaddress.ip_address(value)
        return isinstance(addr, ipaddress.IPv6Address)
    except ValueError:
        return False


def _normalize_host_for_binding(host: str, family: int) -> str:
    """
    Return a host string normalized for use as a socket bind address.
    
    This normalizes whitespace, unwraps IPv6 literals enclosed in square brackets when binding to IPv6,
    and maps empty or unspecified hosts to appropriate bind addresses for the requested address family.
    
    Parameters:
        host (str): The original host string (may contain surrounding whitespace or IPv6 brackets).
        family (int): Address family (e.g., socket.AF_INET or socket.AF_INET6) determining IPv4 vs IPv6 behavior.
    
    Returns:
        str: A host string suitable for binding:
            - For IPv6 family: the unwrapped IPv6 address if bracketed, the original non-empty value otherwise,
              or "::" if the input is empty.
            - For IPv4 family: the original non-empty value, or DEFAULT_BIND_HOST if the input is empty or "::".
    """
    value = host.strip()
    if not value:
        return "::" if family == socket.AF_INET6 else DEFAULT_BIND_HOST
    if family == socket.AF_INET6 and value.startswith("[") and value.endswith("]"):
        return value[1:-1]
    if family == socket.AF_INET6:
        return value
    if value == "::":
        return DEFAULT_BIND_HOST
    return value