import math

class RGBUnit:
    def __init__(self, r=0, g=0, b=0,label = ""):
        if r > 255 or g > 255 or b > 255:
            raise ValueError("Color values must not exceed 255")
        self.r = r
        self.g = g
        self.b = b
        self.label = label


    #Gets the distance between two RGBUnits. This is given in the form of a float.
    def distance_to(self,other):
        rDiff = abs(self.r - other.r) ** 2
        gDiff = abs(self.g - other.g) ** 2
        bDiff = abs(self.b - other.b) ** 2

        return math.sqrt(rDiff + gDiff + bDiff)


unit1 = RGBUnit(100,0,0)
unit2 = RGBUnit(0,0,100)

source = RGBUnit(0,0,0)

source2 = RGBUnit(255,255,255)



print("RGB")