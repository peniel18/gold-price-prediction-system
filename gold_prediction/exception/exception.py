import sys


class CustomException(Exception):
    def __init__(self, error_message, error_detials: sys):
        self.error_message = error_message
        _, _, exc_tb = error_detials.exc_info()
        self.lineNo = exc_tb.tb_lineno # get the line number of the error 
        self.fileName = exc_tb.tb_frame.f_code.co_filename


    def __str__(self):
        return f"Error Occured in python script: [ {self.fileName} ] and on line number: [ {self.lineNo} ]"