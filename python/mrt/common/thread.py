import os
import sys
import logging
import signal
from threading import Event, Thread, Lock

__QUIT_EVENT__ = Event()
""" Maintain primitive for naive array of __QUIT_EVENTS__. """
__LOCK__ = Lock()
__QUIT_EVENTS__ = []

logger = logging.getLogger("service")

# Exit handlers module
__EXIT_HANDLERS__ = []

def register_exit_handler(func):
    __EXIT_HANDLERS__.append(func)
    return func

def safe_exit(*unused_args):
    if __QUIT_EVENT__.is_set():
        logger.warning("Duplicated exit for threads")
        return

    print("")
    logger.info("shutting down ...")

    __LOCK__.acquire()
    __QUIT_EVENT__.set()
    for event in __QUIT_EVENTS__:
        event.set()
    __LOCK__.release()

    for exit_func in __EXIT_HANDLERS__:
        t = Thread(target=exit_func)
        t.join()

# register signal processor in global.
for sig in ('TERM', 'HUP', 'INT'):
    signal.signal(
        getattr(signal, 'SIG'+sig),
        safe_exit);

# Interrupt Design
class ThreadInterruptError(Exception):
    pass

def is_interrupted():
    return __QUIT_EVENT__.is_set()

def interrupt_point():
    if is_interrupted():
        raise ThreadInterruptError()

def wait_for_event(timeout):
    event = Event()

    __LOCK__.acquire()
    if is_interrupted():
        event.set()
    __QUIT_EVENTS__.append(event)
    __LOCK__.release()

    event.wait(timeout)

    __LOCK__.acquire()
    __QUIT_EVENTS__.remove(event)
    __LOCK__.release()

    return event

def wait(timeout):
    """ Thread wait function, raise error if trigger signal int. """
    if wait_for_event(timeout).is_set():
        raise ThreadInterruptError()

def interrupt():
    """ Interrupt program, stop all threads including current. """
    safe_exit()
    interrupt_point()

def _thread_safe_func(func):
    def error_func(*args, **kw):
        try:
            func(*args, **kw)
        except ThreadInterruptError:
            pass
        except Exception as e:
            logger.error("func:{} exit with error: {}".format(
                func.__name__, e))
            safe_exit()
    return error_func

def as_thread_func(func):
    def _container(*args, **kwargs):
        t : Thread = Thread(
                target=_thread_safe_func(func),
                args=args, kwargs=kwargs)
        t.start()
        return t
    return _container

def as_daemon_thread(func):
    """ Daemon Thread Wrapper

        Thread will be closed after main thread exits automatically,
        quick function to programming in development.
    """
    def _container(*args, **kwargs):
        t : Thread = Thread(
                target=_thread_safe_func(func),
                args=args, kwargs=kwargs,
                daemon=True)
        t.start()
        return t
    return _container


# Service module
__REGISTER_SERVICES__ = {}

def register_service(name, auto_reload=False, time_out=5):
    if name in __REGISTER_SERVICES__:
        raise RuntimeError("service:{} has been registered".format(
            name))

    def _func(func):
        def _auto_reload(*args, **kw):
            func(*args, **kw)

            while auto_reload:
                logger.warning(
                    "service:{} closed, restart in {} seconds".format(
                        name, time_out))

                interrupt_point()
                if wait_for_event(time_out).is_set():
                    return

                func(*args, **kw)

        __REGISTER_SERVICES__[name] = as_thread_func(_auto_reload)
        return func
    return _func

def start_services(*args, **kw):
    for name, srv_func in __REGISTER_SERVICES__.items():
        logger.debug("start service - {}".format(name))
        srv_func(*args, **kw)
