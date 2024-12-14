import logging
from queue import Queue, Full

from lada import LOG_LEVEL

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)

def put_closing_queue_marker(queue: Queue, debug_queue_name: str):
    sent_out_none_success = False
    while not sent_out_none_success:
        try:
            queue.put(None, block=False)
            sent_out_none_success = True
        except Full:
            queue.get(block=False)
            queue.task_done()
    logger.debug(f"sent out None to queue {debug_queue_name} to indicate we're stopping")


def empty_out_queue(queue: Queue, debug_queue_name: str):
    while not queue.empty():
        queue.get()
        queue.task_done()
    logger.debug(f"purged all remaining elements from queue {debug_queue_name}")