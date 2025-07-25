import RGB
import csv

class RGB_Graph:
    
    
    #we will use this to read 
    def __init__(self,filepath):
        pass

    def digest_unclassified_RGB(self,filepath):
        pass

    def digest_csv(self,filepath):
        data = []
        with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)

            for row in csv_reader:
                data.append(row)
                print(row)
                x = input("test")


unit = RGB_Graph("test")

unit.digest_csv("colors.csv")