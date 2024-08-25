import os
from typing import Any, Union
import numpy as np
import pandas as pd

#################################################################################
# heading utils 
#################################################################################
def breakline(char: str='-') -> str:
    """
        --- Purpose ---
        Returns a string of characters {char} that fill the entire console width

        

        --- Parameters ---
        char(str): a single character that will be used the create the line

        

        --- Return Type ---
        str


        
        --- Examples ---
        print(breakline())

        print(breakline("*"))

        try:
            print(breakline(None))
        except TypeError:
            print("You cannot set the char parameter to the value  of None")

        print(breakline(1))

        print(breakline("*-"))

        print(breakline(" "))

        print(breakline("+"))
    """
    return char * os.get_terminal_size().columns


def heading(text: str, char: str='-', full: bool=False) -> str:
    """
        --- Purpose ---
        Creates a string where {text} is centered followed by a line of characters {char}
        that fill the width of the console

        

        --- Parameters ---
        text(str): a string of text that you want for your header

        char(str): a character that will be used to create the line break
        -   default value: "-"

        full(bool): a boolean that indicates where there should be a line at the
        top of the heading or not

        

        --- Return Type ---
        str - a string of text for the header

        

        --- Examples ---
        print(heading("This is a heading"))

        print(heading("This is another heading", "*"))

        try:
            print(heading(None))
        except AttributeError:
            print("The text parameter cannot be of type NoneType")

        try:
            print(heading("An error will occur", None))
        except TypeError:
            print("The char parameter cannot be of type NoneType")

        print(heading("This is a heading with a thicker line", "--"))

        print(heading(""))
        
        print(heading("", full=True))

        print(heading("A final heading", full=True))
    """
    cw = os.get_terminal_size().columns
    line = cw*char
    first_line = line + "\n" if full else ""
    return f"{first_line}{text.center(cw)}\n{line}"


def module_intro(module_name: str):
    print(breakline())
    print(heading(f"In {module_name}"))


#################################################################################
# array and string manipulation utils
#################################################################################
def arr_merge(arr1: Union[list[Any], np.array], arr2: Union[list[Any], np.array]) -> np.array:
    """
        --- Purpose ---
        Takes two arrays and picks items from them alternately to combine into a single
        array



        --- Examples ---
        print(arr_merge([1, 2, 3, 4], np.array([0, 0, 0, 0])))

        print(arr_merge(np.array([1, 2, 3, 4]), np.array(["A", "B", "C", "D"])))
    """
    n1: int = len(arr1)
    n2: int = len(arr2)
    n: int = max(n1, n2)
    # I would love to know a more efficient way of doing this.
    # Using np.empty creates errors, presumably due to string length and memory allocation.
    result = np.concatenate((arr1, arr2))
    index: int = 0
    for i in range(n):
        if i < n1:
            result[index] = arr1[i]
            index += 1
        if i < n2:
            result[index] = arr2[i]
            index += 1
    return result



def str_merge(str_arr1: Union[list[str], np.array],
              str_arr2: Union[list[str], np.array],
              delim: str="") -> str:
    """
        --- Purpose ---
        Takes two arrays of strings and picks items from the arrays alternately to combine
        into a single string

        

        --- Parameters ---
        str_arr1(Union[list[str], np.array): The first string array you want to merge

        str_arr2(Union[list[str], np.array): The second string array you want to merge

        delim(str)="": A string delimeter that you want between each individual string
        that you merge



        --- Return Type ---
        str: a string of the merged arrays



        --- Examples ---
        str_arr1 = ["a", "b", "c"]
        str_arr2 = ["1", "2", "3"]
        print(str_merge(["a", "b", "c"], ["1", "2", "3"]))

        str_arr1 = np.array(str_arr1)
        str_arr2 = np.array(str_arr2)
        print(str_merge(str_arr1, str_arr2))

        print(str_merge(str_arr1, str_arr2, "-"))
    """
    return delim.join(arr_merge(str_arr1, str_arr2))


#################################################################################
# matrix functions
#################################################################################
def dframe_to_intmatrix(dframe: pd.DataFrame, vars: list[str]=None, ones: bool=True) -> np.array:
    """
        --- Purpose ---
        Prepares a data frame to be used as a matrix in linear regression by ordering the variables
        appropiately into the right columns and adding a column of 1's to the beginning of the
        matrix.

        

        --- Parameters ---
        dframe(pd.DataFrame): a pandas DataFrame that supplies the data for the matrix

        vars(list[str]): a list of strings for the names of the columns in the order that
        you want them for the resulting matrix

        ones(bool): a boolean indicating whether or not a column of 1's should be added to
        the beginning of the matrix
        -   default: the default value is True and will create a column of 1's in the 0 column
            position for the resulting matrix


        
        --- Return Type ---
        np.array[float]: a 2-dimensional numpy array of floats



        --- Examples ---
        data = pd.DataFrame({"x1": [1, 2, 3, 4], "x2": [1, 4, 9, 16]})
        print(dframe_to_intmatrix(data))
        print(dframe_to_intmatrix(data, ["x2", "x1"]))
        print(dframe_to_intmatrix(data, ones=False))
    """
    if not vars:
        vars = dframe.columns
    if ones:
        n = dframe.shape[0]
        return pd.concat((pd.DataFrame(np.ones(n)), dframe[vars]), axis=1).to_numpy()
    return dframe[vars].to_numpy()

def array_to_intmatrix(arr: np.array) -> np.array:
    """
        --- Purpose ---
        Adds a column of 1's to the beginning of a 2-D numpy array for use in certain
        calculations.

        --- Parameters ---
        arr(np.array[float]): a numpy array of floats

        --- Return Type ---
        np.array[float]: a numpy array of floats

        --- Example ---
        print(array_to_intmatrix(np.array([1, 2, 3, 4]).reshape((2, 2))))
    """
    n: int = arr.shape[0]
    return np.concatenate((np.ones(n).reshape((n, 1)), arr), axis=1)

#################################################################################
# testing
#################################################################################
def main():
    module_intro("util.py")
    print(array_to_intmatrix(np.array([1, 2, 3, 4]).reshape((2, 2))))
    data = pd.DataFrame({"x1": [1, 2, 3, 4], "x2": [1, 4, 9, 16]})
    print(dframe_to_intmatrix(data))
    print(dframe_to_intmatrix(data, ["x2", "x1"]))
    print(dframe_to_intmatrix(data, ones=False))
    print(arr_merge([1, 2, 3, 4], np.array([0, 0, 0, 0])))

if __name__ == "__main__":
    main()