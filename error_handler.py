import logging
logger = logging.getLogger('ErrorHandler')
import traceback
import sys

from PyQt5.QtWidgets import QMessageBox

def error_handler(exception, context=None, messagebox=True):
    print(traceback.format_exc(), file=sys.stderr, flush=True)
    if context is None:
        context = '(unspecified)'

    if messagebox:
        msgbox = QMessageBox()
        msgbox.setText(
            f'The following error occurred while trying to perform the operation \"{context}\":<br />'
            f'<pre>{str(exception)}</pre><br />'
            f'See the terminal output for traceback and other information'
        )
        msgbox.exec()
    else: # Display on console
        silent_error_handler(exception=exception, context=context)

def silent_error_handler(exception, context=None):
    logger.error('An exception occurred while trying to perform the operation \"{}\"'.format(
        context if context is not None else '(unspecified)'
    ))
    print(traceback.format_exc(), file=sys.stderr, flush=True)
