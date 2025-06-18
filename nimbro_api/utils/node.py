#!/usr/bin/env python3

import os
import time
import threading
import traceback

import rclpy
from rclpy.executors import MultiThreadedExecutor

class Colors:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class SelfShutdown(Exception):
    pass

def block_until_future_complete(node, future, timeout_sec=None):
    condition = threading.Condition()
    done_flag = [False]

    def future_done_cb(_):
        with condition:
            done_flag[0] = True
            condition.notify_all()

    future.add_done_callback(future_done_cb)

    start_time = time.monotonic()

    with condition:
        while not done_flag[0]:
            time_left = None
            if timeout_sec is not None:
                elapsed = time.monotonic() - start_time
                time_left = max(0.0, timeout_sec - elapsed)
                if time_left == 0.0:
                    return False

            try:
                node.executor.spin_once(timeout_sec=time_left)
            except KeyboardInterrupt:
                os._exit(0)
                # raise KeyboardInterrupt
            except Exception as e:
                if node.executor is None:
                    node.get_logger().error(f"{repr(e)}\n{traceback.format_exc()}\n\nTry using the other block_until_future_complete function below, by uncommenting it and commenting this")
                elif isinstance(e, ValueError) and 'generator already executing' in str(e):
                    pass # someone else is already spinning this executor
                else:
                    if not rclpy.ok():
                        os._exit(0)
                        # raise e
                    node.get_logger().error(f"{repr(e)}\n{traceback.format_exc()}")
                    if timeout_sec is not None and (time.monotonic() - start_time) >= timeout_sec:
                        return False

            condition.wait(timeout=0.01)

    return True

# def block_until_future_complete(node, future, timeout_sec=None):
#     if not hasattr(node, 'is_spinning'):
#         rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)
#     elif not node.is_spinning:
#         rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)
#     else:
#         event = threading.Event()

#         def unblock(future):
#             nonlocal event
#             event.set()

#         future.add_done_callback(unblock)

#         if not future.done():
#             event.wait(timeout=timeout_sec)
#         if future.exception() is not None:
#             raise future.exception()

def start_and_spin_node(node_cls, *, args=None, node_args=None, num_threads=100, blocking=True, os_shutdown=False):
    """
    Instantiate and spin a node with a MultiThreadedExecutor, with robust
    exception handling and optional forced exit in blocking mode.

    :param node_cls:    Your Node class (subclass of rclpy.node.Node).
    :param args:        Arguments passed to context.init(args=...).
    :param node_args:   Optional dict passed to node_cls(**node_args).
    :param num_threads: Passed to MultiThreadedExecutor(num_threads=...), which uses the CPU if None.
    :param blocking:    If True, spins in this thread (blocks) and auto-cleans.
                        If False, spins in a daemon thread and returns immediately—
                        you must manually call executor.shutdown(), node.destroy_node(),
                        and context.shutdown() when you’re done.
    :param os_shutdown: If True, calls os._exit(0) at the very end of all paths.
    :return:            If blocking=False, returns (node, executor, context, thread). Else None.
    """

    # Create an explicit shared context and initialize it
    context = rclpy.Context()
    context.init(args=args)
    if blocking:
        rclpy.signals.install_signal_handlers(rclpy.signals.SignalHandlerOptions.ALL)

    # Instantiate the node with the shared context
    try:
        node = node_cls(context=context, **(node_args or {}))
    except KeyboardInterrupt:
        print(f"{Colors.DARKCYAN}Node interrupted{Colors.END}")
        return
    except SelfShutdown as e:
        context.try_shutdown()
        if not blocking:
            raise e
        print(f"{Colors.GREEN}Node triggered self shutdown{'' if str(e) == '' else (': ' + str(e))}{Colors.END}")
        return
    except Exception as e:
        context.try_shutdown()
        if not blocking:
            raise e
        trace = traceback.format_exc()
        print(f"{Colors.RED}Exception occurred while initializing node: {repr(e)}{Colors.END}")
        print(f"{Colors.RED}{trace}{Colors.END}")
        return

    # Create executor with same context and add node
    executor = MultiThreadedExecutor(context=context, num_threads=num_threads)
    added = executor.add_node(node)
    assert added, (
        f"Failed to add node with context {node.context!r} "
        f"to executor with context {executor.context!r}"
    )

    # Define spin logic
    def _spin_and_handle():
        try:
            executor.spin()
        except KeyboardInterrupt:
            print(f"{Colors.DARKCYAN}Node interrupted{Colors.END}")
        except SelfShutdown as e:
            print(f"{Colors.GREEN}Node triggered self shutdown{'' if str(e) == '' else (': ' + str(e))}{Colors.END}")
        except Exception as e:
            trace = traceback.format_exc()
            print(f"{Colors.RED}Exception occurred while spinning node: {repr(e)}{Colors.END}")
            print(f"{Colors.RED}{trace}{Colors.END}")

    # Run
    if blocking:
        try:
            _spin_and_handle()
        finally:
            executor.shutdown()
            node.destroy_node()
            if os_shutdown:
                print(f"{Colors.YELLOW}Forcing ungraceful node shutdown{Colors.END}")
                os._exit(0)
            context.try_shutdown()
    else:
        thread = threading.Thread(target=_spin_and_handle, daemon=True)
        thread.start()
        return node, executor, context, thread
