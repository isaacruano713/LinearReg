from typing import Self
import utils as u

class LinearForm:
    """
        --- Purpose ---
        LinearForm objects are formulas that can be used to create LinearReg and similar
        objects. They are used to keep track of the relationship between the dependent
        ("y") and indepdendent ("x") variables. They also show a string representation of
        this relationship.

        

        --- Properties ---
        formula(str): a string representation of the formula

        x(list[str]): a list of the names of the independent ("x") variables

        y(str): a string with the name of the dependent ("y") variable

        is_pretty(bool): a boolean indicating whether the formula has been "prettified" or not



        --- Methods ---
        Instance Methods
            __init__(self, formula: str) -> self
            -   initializes a LinearForm object

            __repr__(self) -> str
            -   returns a string representation of a LinearForm object for printing

            prettify(self) -> str
            -   reformats the formula property to be neater


        Class Methods
            from_list(x: list[str], y: str) -> Self
            -   an alternative way of initializing a LinearForm using a list of x variables
                and a y variable



        --- Examples ---
        # The basic way of writing these formulas is as follows:
        #     LinearForm("y = x1 + x2 + x3 + ...")
        # where y is the name of the dependent variable used in the linear regression model,
        # and x1, x2, x3, ... are all of the dependent variables for the model.

        # Keep in mind that the names used for y, x1, x2, ..., should all match the column
        # names in the DataFrame used to create the model.

        print(LinearForm("money = time + effort"))

        print(LinearForm("y = x1 + x2 + x3"))

        print(LinearForm.from_lists(["x1", "x2"], "y"))
    """


    def __init__(self, formula: str) -> Self:
        self.formula: str = formula
        vars: list[str] = formula.split("=")
        self.y: str = vars[0].strip()
        self.x: list[str] = [x.strip() for x in vars[1].split('+')]
        self.is_pretty: bool = False

    def prettify(self) -> None:
        """
            --- Purpose ---
            Reformats the formula to make it look neater.

            A formula like y=x+x2 will then become y = x + x2.
        """
        if not self.is_pretty:
            self.formula = f"{self.y} = {" + ".join(self.x)}"
            self.is_pretty = True

    def __repr__(self) -> None:
        if not self.is_pretty:
            self.prettify()
        return f"{u.heading("LinearForm")}\nFormula: {self.formula}\nx: {self.x}\ny: {self.y}"

    @classmethod
    def from_list(cls, x: list[str], y: str) -> Self:
        """
            --- Purpose ---
            This is an alternative method of creating a LinearForm object if it is more convenient
            to supply the names of the variables in a list that it is to write out the formula.
            
            --- Example ---
            LinearForm.from_list(["x1", "x2", "x3"], "y")
            LinearForm.from_list(["carrot", "celery", "onion", "water"], "soup")
        """
        return LinearForm("{}={}".format(y, "+".join(x)))
    
def main():
    u.module_intro("LinearForm.py")
    print(LinearForm.from_list(["x1", "x2", "x3"], "y"))
    print(LinearForm.from_list(["carrots", "celery", "onion", "water"], "soup"))
    

if __name__ == "__main__":
    main()
