# exceptional handling module for a Python application
# This module defines a custom exception class that captures detailed error information.
import sys

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    error_message = "Error occurred in script [{0}] at line number [{1}] with message [{2}]".format(
        exc_tb.tb_frame.f_code.co_filename, exc_tb.tb_lineno, str(error)
    )
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message

    def __repr__(self):
        return str(self)



# if __name__ == "__main__":
#     try:
#         a=1/0  # This will raise a ZeroDivisionError
#     except Exception as e:
#         from src.logger import logging  # Assuming logger is defined in src/logger.py
#         logging.info("divide by zero error")
#         raise CustomException(e, sys)
#         # This will raise a CustomException with detailed error information
