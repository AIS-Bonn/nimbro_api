#!/usr/bin/env python3

import os
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
    if not hasattr(node, 'is_spinning'):
        rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)
    elif not node.is_spinning:
        rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)
    else:
        event = threading.Event()

        def unblock(future):
            nonlocal event
            event.set()

        future.add_done_callback(unblock)

        if not future.done():
            event.wait(timeout=timeout_sec)
        if future.exception() is not None:
            raise future.exception()

def spin_executor(executor):
    try:
        executor.spin()
    except rclpy.executors.ExternalShutdownException:
        print("External Shutdown Request!")

def spin_node_with_multi_threaded_executor(node, blocking=True):
    executor = MultiThreadedExecutor()
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
    except SelfShutdown:
        print("Node triggered self shutdown")
    except Exception as e:
        trace = traceback.format_exc()
        print(f"{Colors.RED}Exception occurred while initializing node" + (f": {e}" if str(e) != '' else '') + f"{Colors.END}")
        print(f"{Colors.RED}{trace}{Colors.END}")
    else:
        try:
            spin_node_with_multi_threaded_executor(node)
        except KeyboardInterrupt:
            node.destroy_node()
            print("Node interrupted")
        except SelfShutdown:
            node.destroy_node()
            print("Node triggered self shutdown")
        except Exception as e:
            node.destroy_node()
            trace = traceback.format_exc()
            node.get_logger().error("Node crashed after Exception" + (f": {e}" if str(e) != '' else ''))
            node.get_logger().error(trace)
    if os_shutdown:
        print("Forcing ungraceful node shutdown")
        os._exit(0)
