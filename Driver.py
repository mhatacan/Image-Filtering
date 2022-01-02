###########################################
# Filename    :  Driver.py
# Author      :  Muhammet Harun ATACAN
# Date        :  13.06.2021
# Description :  Driver Code
###########################################

import Operation
import sys
from PyQt5 import QtWidgets

def main():
    
    app = QtWidgets.QApplication(sys.argv)
    pencere = Operation.Operation()
    pencere.show()
    sys.exit(app.exec_())

main()