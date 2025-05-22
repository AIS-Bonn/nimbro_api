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
                if isinstance(e, ValueError) and 'generator already executing' in str(e):
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

def spin_executor(executor):
    try:
        executor.spin()
    except rclpy.executors.ExternalShutdownException:
        print("External Shutdown Request!")

def spin_node_with_multi_threaded_executor(node, blocking=True):
    executor = MultiThreadedExecutor(num_threads=100)
    executor.add_node(node)
    node.is_spinning = True

    if blocking:
        spin_executor(executor)
    else:
        executor_thread = threading.Thread(target=spin_executor, args=(executor, ), daemon=True)
        executor_thread.start()

def start_and_spin_node(node_class, args=None, node_args=None, os_shutdown=False):
    rclpy.init(args=args)
    try:
        if node_args is None:
            node = node_class()
        else:
            node = node_class(**node_args)
    except KeyboardInterrupt:
        print("Node interrupted")
    except SelfShutdown as e:
        msg = str(e)
        print(f"Node triggered self shutdown{(': ' + msg) if msg != '' else ''}")
    except Exception as e:
        trace = traceback.format_exc()
        print(f"{Colors.RED}Exception occurred while initializing node" + (f": {repr(e)}" if repr(e) != '' else '') + f"{Colors.END}")
        print(f"{Colors.RED}{trace}{Colors.END}")
    else:
        try:
            spin_node_with_multi_threaded_executor(node)
        except KeyboardInterrupt:
            node.destroy_node()
            print("Node interrupted")
        except SelfShutdown as e:
            node.destroy_node()
            msg = str(e)
            print(f"Node triggered self shutdown{(': ' + msg) if msg != '' else ''}")
        except Exception as e:
            node.destroy_node()
            trace = traceback.format_exc()
            node.get_logger().error("Node crashed after Exception" + (f": {repr(e)}" if repr(e) != '' else ''))
            node.get_logger().error(trace)
    if os_shutdown:
        print("Forcing ungraceful node shutdown")
        os._exit(0)
