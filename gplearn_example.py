import numpy as np
from matplotlib import pyplot as plt
from gplearn.genetic import SymbolicRegressor
import graphviz

# Make some data
N = 10
x = np.linspace(0, 10, N)
y = 2 + x**2

# Initialise symbolic regression code
est_gp = SymbolicRegressor(population_size=100,
                           generations=20,
                           function_set=('add', 'sub', 'mul', 'div'),
                           verbose=1,
                           parsimony_coefficient=0.01)

# Train and print model
est_gp.fit(np.vstack(x), y)
print(est_gp._program)

# Predictions
y_pred = est_gp.predict(np.vstack(x))

# Plot graph - not working at the moment!
'''
dot_data = est_gp._program.export_graphviz()
graph = graphviz.Source(dot_data)
graph.view()
'''

# Plot predictions
fig, ax = plt.subplots()
ax.plot(x, y, 'o')
ax.plot(x, y_pred)
plt.show()
