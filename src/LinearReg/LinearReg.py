from typing import Union
import numpy as np
import pandas as pd
import util as u
from LinearForm import LinearForm


class LinearReg:
    """
    --- Purpose ---
    LinearReg objects are used to fit and analyze linear models.

    

    --- Properties ---
    formula(LinearForm): LinearForm object that shows the dependent ("y") and
    independent ("x") variables of the LinearReg object

    y(np.array[float]): 1-dimensional numpy array containing the data for the
    dependent variable

    x(np.array[float]): 2-dimensional numpy array containing the data for the
    independent variables
    -   includes a column of 1's appended to the beggining of the array

    yvar(str): string that holds the name of the dependent variable

    xvars(list[str]): a list of strings containing the names of the independent
    variables

    n(int): an integer tracking the number of rows/observations in the data
    -   equal to the length of y or the length of the first dimension of x

    k(int): an integer that tracks the number of independent ("x") variables

    B(np.array[float]): 1-dimensional numpy array containing the parameter
    estimates for the linear regression model

    model(str): a string representation of the fitted linear model



    --- Methods ---
    Instance Methods
        __init__(self) -> Self
        -   initializes the LinearReg object

        __repr__(self) -> str
        -   creates a string representation of the LinearReg object when
            printing

        write_model(self) -> None
        -   creates a string representation of the fitted linear model

        fit(self, formula: str | LinearForm, dframe: pd.DataFrame) -> None
        -   fits the linear model to the data

        pred(self, data=self.x) -> np.array[float]
        -   uses the linear model to return model predictions
        -   if no arguments are passed, it returns the predictions using the
            data provided to fit the model
        
        resid(self, data=self.x) -> np.array[float]
        -   uses the linear model to calculate residuals for model predictions
        -   if no arguments are passed, it returns the residuals using the data
            provided to fit the model

        test(self, alt: list[LinearForm]) -> None
        -   tests the significance of the parameters in the model
        -   prints the information it finds
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
        --- Purpose ---
        Internal function that is used to write a representation of the linear
        model. If model is None then the function will create this
        representation using the parameter estimates and the variable names.
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
            self.model = f"{self.yvar} = {u.str_merge(xvars, signs)}"

    ############################################################################
    # Here are all the methods that involve actual data analysis
    ############################################################################
    # def transform(self):

    def fit(self,
            formula: Union[str, LinearForm],
            dframe: pd.DataFrame) -> None:
        """
        --- Purpose ---
        Fits the linear model to the data

        

        --- Parameters ---
        formula: either a string or a LinearForm that indicates what the
        dependent ("y") varaiable of the model is and what the independent ("x")
        variables of the model are
        -   Ex. "y = x1 + x2" or LinearForm("y = x1 + x2")

        dframe: a pandas DataFrame containing the data that will be used for
        the model

        

        --- Return Type ---
        None



        --- Examples ---
        data = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8],
            "y": [2, 5, 4, 9, 4, 7, 15, 11]})
        lm = LinearReg()

        lm.fit("y = x", data)
        print(lm)

        lm.fit(LinearForm("y = x"), data)
        print(lm)
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
        self.x = u.dframe_to_intmatrix(dframe, self.xvars)
        self.model = None

        xtx = np.transpose(self.x) @ self.x
        C = np.linalg.inv(xtx)
        B = C @ np.transpose(self.x) @ self.y

        self.xtx = xtx
        self.C = C
        self.B = B

################################################################################
    def pred(self,
             data: Union[np.array, list[float],
             pd.DataFrame, None]=None) -> np.array:
        """
        --- Purpose ---
        Uses the linear model to return model predictions
        If no arguments are passed, it returns the predictions using the data
        provided to fit the model.
        


        --- Parameters ---
        data(Union[np.array, list[float], pd.DataFrame, None]): a 2-dimensional
        numpy array, a list of floats, or a pandas DataFrame containing the data
        that you want to make predictions on
        -   For all of the options supplied for {data}, do not include an
            intercept value, or column; an intercept value or column will
            automatically be added
        -   If {data} is a numpy array, it must be 2-dimensional, it must have
            all of its columns in the same order as they are in the formula, and
            it must include only the columns for the independent ("x")
            variables.
        -   If {data} is a pandas DataFrame, make sure it has a column for each
            of the independent variables, and make sure that the column names
            match up with the names of the variables in {self.xvars}, but the
            columns do not need to be in the same order as {self.xvars}
        -   If {data} is a list, it must be a 1-dimensional list used to
            calculate only one prediction. It must also contain values for all
            of the dependent variables in the order that they are presented in
            {self.xvars}
        -   default value: None

        

        --- Return Type ---
        np.array[float]



        --- Examples ---
        data = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8],
            "y": [2, 5, 4, 9, 4, 7, 15, 11]})

        lm = LinearReg()
        lm.fit("y = x", data)
        print(lm)
        print(lm.pred())
        print(lm.pred(pd.DataFrame({"x":[9]})))
        print(lm.pred([9]))
        try:
            print(lm.pred(9))
        except TypeError:
            print("You cannot enter a simple data type. You must use a \\
list, array, or DataFrame.")
        print(lm.pred(np.array([5, 6, 7, 8]).reshape((4, 1))))
        """
        if isinstance(data, np.ndarray):
            return u.array_to_intmatrix(data) @ self.B
        if isinstance(data, list):
            data.insert(0, 1)
            return np.array([np.dot(data, self.B)])
        if isinstance(data, pd.DataFrame):
            return u.dframe_to_intmatrix(data, self.xvars) @ self.B
        if not data:
            return self.x @ self.B
        raise TypeError("Parameter data is of an unidentified type. Must be \
a list, DataFrame or numpy array.")

    def resid(self,
              data: Union[np.array, list[float], pd.DataFrame, None]=None,
              y: Union[np.array, list[float]]=None) -> np.array:
        """
        --- Purpose ---
        Calculates and returns the residuals of a linear model. Will return the
        residuals of the data in {self.x} if no other data is specified.
        


        --- Parameters ---
        data(Union[np.array, list[float], pd.DataFrame, None]): the data that
        you supply to calculate the residuals
        -   defaults to {self.x} if the value is set to None
        -   see {self.pred()} to learn other properties for this parameter

        y(Union[np.array[float], list[float]]): a list or array of floats for
        the true values for the dependent variable of the linear model



        --- Return Type ---
        np.array[float]: an array of floats for the calulated residuals



        --- Examples ---
        data = pd.DataFrame({"x1": [1, 2, 3, 4, 5, 6, 7, 8],
                             "x2": [5, 23, 13, 23, 33, 12, 34, 23],
                             "y": [3, 2, 3, 5, 3, 6, 11, 2]})

        lm = LinearReg()
        lm.fit("y = x1 + x2", data)

        print(lm)
        print(lm.resid())
        print(lm.resid([2, 9], [2]))
        print(lm.resid(pd.DataFrame({"x1": [2, 3, 4], "x2": [39, 12, 48]}),
                       y=[5, 3, 6]))
        print(lm.resid(np.array([[2, 39], [3, 12], [4, 48]]),
                       y=np.array([5, 3, 6])))
        """
        if y is None:
            y = self.y
        if isinstance(y, list):
            y = np.array(y)
        predictions = self.pred(data)
        return y - predictions

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
        return "\n".join([u.heading("LinearReg"),
                          formula_str,
                          model_str,
                          xvars_str,
                          yvar_str,
                          n_k_str,
                          ydata_str,
                          xdata_str,
                          B_str])

################################################################################
# Testing
################################################################################
def main():
    print("#"*80)

if __name__ == "__main__":
    main()
