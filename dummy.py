import json
import numpy as np
import hyper_parameter_providers as hpp

print(hpp.ConstantParameterProvider.__name__)
d = {
    "a":[i for i in range(10)],
    "b" : 20.5,
    "c" : ["apple", "orange"]
}

print(json.dumps(d))
with open("./json.json", "w") as outputfile:
    json.dump(d, outputfile)

with open("./json.json", "r") as inputfile:
    d2 = json.load(inputfile)

print(d2)