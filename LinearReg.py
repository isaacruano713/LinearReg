import os
from typing import Any, Self, Union
import numpy as np
import pandas as pd

def breakline(char: str='-') -> str:
    return char * os.get_terminal_size().columns


def heading(text: str, char: str='-') -> str:
    """
        Creates a string where {text} is centered followed by a line of characters {char}
        that fill the width of the console

        Try: print(heading("chicken"))
    """
    cw = os.get_terminal_size().columns
    return f"{text.center(cw)}\n{cw*char}"

def arr_merge(arr1: Union[list[Any], np.array], arr2: Union[list[Any], np.array]) -> np.array:
    """
        Takes two arrays and picks items from them alternately to combine into a single
        array
    """
    n1: int = len(arr1)
    n2: int = len(arr2)
    n: int = max(n1, n2)
    N: int = n1 + n2
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
              delim: str="") -> np.array:
    """
        Takes two arrays of strings and picks items from the arrays alternately to combine
        into a single string
    """
    return delim.join(arr_merge(str_arr1, str_arr2))


class LinearForm:
    """
        LinearForm objects are formulas that can be used to create LinearReg and similar
        objects.

        The basic way of writing these formulas is as follows:
            LinearForm("y = x1 + x2 + x3 + ...")
        where y is the name of the dependent variable used in the linear regression model,
        and x1, x2, x3, ... are all of the dependent variables for the model.

        Keep in mind that the names used for y, x1, x2, ..., should all match the column
        names in the DataFrame used to create the model.

        Try:
            LinearForm("money = time + effort")
    """


    def __init__(self, formula: str):
        self.formula = formula
        vars = formula.split("=")
        self.y = vars[0].strip()
        self.x = [x.strip() for x in vars[1].split('+')]
        self.is_pretty = False

    def prettify(self) -> None:
        """
            Reformats the formula to make it look neater.

            A formula like y=x+x2 will then become y = x + x2.
        """
        if not self.is_pretty:
            self.formula = f"{self.y} = {" + ".join(self.x)}"
            self.is_pretty = True

    def __repr__(self) -> None:
        if not self.is_pretty:
            self.prettify()
        return f"{heading("LinearForm")}\nFormula: {self.formula}\nx: {self.x}\ny: {self.y}"

    @staticmethod
    def from_list(x: list[str], y: str) -> Self:
        """
            This is an alternative method of creating a LinearForm object if it is more convenient
            to supply the names of the variables in a list that it is to write out the formula.
            
            Try: LinearForm.from_list(['x1', 'x2', 'x3'], 'y')
        """
        return LinearForm("{}={}".format(y, "+".join(x)))


class LinearReg:
    """
        --- Purpose ---
        LinearReg objects are used to fit and analyze linear models.

        

        --- Properties ---
        formula(LinearForm): LinearForm object that shows the dependent ("y") and independent ("x")
        variables of the LinearReg object

        y(np.array[float]): 1-dimensional numpy array containing the data for the dependent variable

        x(np.array[float]): 2-dimensional numpy array containing the data for the independent variables
        -   includes a column of 1's appended to the beggining of the array

        yvar(str): string that holds the name of the dependent variable

        xvars(list[str]): a list of strings containing the names of the independent variables

        n(int): an integer tracking the number of rows/observations in the data
        -   equal to the length of y or the length of the first dimension of x

        k(int): an integer that tracks the number of independent ("x") variables

        B(np.array[float]): 1-dimensional numpy array containing the parameter estimates
        for the linear regression model

        model(str): a string representation of the fitted linear model



        --- Methods ---
        write_model(self) -> None
        -   creates a string representation of the fitted linear model

        fit(self, formula: str | LinearForm, dframe: pd.DataFrame) -> None
        -   fits the linear model to the data

        pred(self, data=self.x) -> np.array[float]
        -   uses the linear model to return model predictions
        -   if no arguments are passed, it returns the predictions using the data provided
            to fit the model
        
        resid(self, data=self.x) -> np.array[float]
        -   uses the linear model to calculate residuals for model predictions
        -   if no arguments are passed, it returns the residuals using the data provided
            to fit the model
    """
    def __init__(self):
        self.formula = None
        self.y = None
        self.yvar = None
        self.x = None
        self.xvars = None
        self.n = None
        self.k = None
        self.B = None
        self.model = None

    def write_model(self) -> None:
        """
            Internal function that is used to write a representation of the linear model. If model is None then
            the function will create this representation using the parameter estimates and the variable names.
            Otherwise 
        """
        if self.formula:
            signs = [" + " if x >= 0 else " - " for x in self.B[1:]]
            B_chars = [str(round(abs(x), 4)) for x in self.B]
            if self.B[0] < 0:
                B_chars[0] = "-" + B_chars[0]
            x_chars = np.char.add("(", self.xvars)
            x_chars = np.char.add(x_chars, ")")
            x_chars = np.concatenate((np.array([""]), x_chars))
            xvars = np.char.add(B_chars, x_chars)
            self.model = f"{self.yvar} = {str_merge(xvars, signs)}"

    ##############################################################
    # Here are all the methods that involve actual data analysis #
    ##############################################################
    # def transform(self):

    def fit(self, formula: Union[str, LinearForm], dframe: pd.DataFrame) -> None:
        """
            --- Purpose ---
            Fits the linear model to the data

            

            --- Parameters ---
            formula: either a string or a LinearForm that indicates what the dependent ("y") varaiable
            of the model is and what the independent ("x") variables of the model are.
            -   Ex. "y = x1 + x2" or LinearForm("y = x1 + x2")

            pdframe: a pandas DataFrame containing the data that will be used for the model

            

            --- Return Type ---
            None



            --- Examples ---











        """
        if type(formula) == str:
            formula = LinearForm(formula)
        formula.prettify()
        self.formula = formula.formula
        self.yvar = formula.y
        self.xvars = formula.x
        self.k = len(self.xvars)
        self.n = dframe.shape[0]
        self.y = dframe[self.yvar].to_numpy()
        self.x = pd.concat((pd.DataFrame(np.ones(self.n, dtype=int)), dframe[self.xvars]), axis=1).to_numpy()
        self.model = None

        xtx = np.transpose(self.x) @ self.x
        C = np.linalg.inv(xtx)
        B = C @ np.transpose(self.x) @ self.y

        self.xtx = xtx
        self.C = C
        self.B = B

    def pred(self, data: np.array=None) -> np.array:
        """
            --- Purpose ---
            Uses the linear model to return model predictions
            If no arguments are passed, it returns the predictions using the data provided
            to fit the model.
            

            --- Parameters ---
            data: a 2-dimensional numpy array containing the data you want to make your predictions
            -   make sure that the column names match the dependent variables
            -   make sure that you have an additional column of 1's in the 0 column position and that
                you have a column for every independent ("x") variable in addition to the column of 1's

            

            --- Return Type ---
            np.array[float]



            --- Examples ---
            import pandas as pd
            import numpy as np
            data = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8], "y": [2, 5, 4, 9, 4, 7, 15, 11]})

            lm = LinearReg()
            lm.fit("y = x", data)
            print(lm.y)
            print(lm.pred())
            print(lm.pred(pd.DataFrame({0: [1], "x":[9]})))
        """
        if data:
            return data @ self.B
        return self.x @ self.B

    def __repr__(self) -> str:
        if not self.model:
            self.write_model()

        formula_str = f"Formula: {self.formula}"
        model_str = f"Model: {self.model}"
        xvars_str = f"xvars: {self.xvars}"
        yvar_str = f"yvar: {self.yvar}"
        n_k_str = f"n: {self.n}, k: {self.k}\n"
        ydata_str = f"ydata:\n{self.y}\n"
        xdata_str = f"xdata:\n{self.x}\n"
        B_str = f"B: {self.B}"
        return "\n".join([heading("LinearReg"),
                          formula_str,
                          model_str,
                          xvars_str,
                          yvar_str,
                          n_k_str,
                          ydata_str,
                          xdata_str,
                          B_str])
    
hi = pd.DataFrame({"time": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                   "money": [123, 283, 230, 304, 420, 293, 394, 392, 500, 512],
                   "effort": [1, 2, 3, 2, 3, 2, 3, 1, 1, 2]})

lm = LinearReg()
lm.fit("money = time + effort", hi)

print(lm.pred())