import matplotlib.pyplot as plt
import numpy as np
from GP.gpr import GaussianProcessRegression


def plot(model: GaussianProcessRegression, bounds, fixed_values=None, label_x=None, label_y=None, filename=None,
         ax=plt):
    '''
    Plots of the model and the acquisition function in 1D and 2D examples.
    '''

    X_train, y_train = model.get_data()
    input_dim = X_train.shape[1]

    if fixed_values is None:
        fixed_values = [None] * input_dim

    free_dimensions = []
    for i, value in enumerate(fixed_values):
        if value is None:
            free_dimensions.append(i)

    assert len(free_dimensions) == 2 or len(free_dimensions) == 1, \
        "In case of dimension being more than 2 fixed_values must " \
        "be specified so as to make a number of free dimensions equal 2 or 1."

    if len(free_dimensions) == 1:
        index_x = free_dimensions[0]

        if not label_x:
            label_x = f'X{index_x + 1}'

        X_values = np.arange(bounds[index_x][0], bounds[index_x][1], 0.001)  # grid
        m = len(X_values)
        X = np.zeros((m, input_dim))
        for i, value in enumerate(fixed_values):
            if value is not None:
                X[:, i] = [value] * m
        X[:, index_x] = X_values
        X_values = X_values.reshape((m, 1))

        m, v = model.predict(X)
        ax.plot(X_values, m, 'b-', label=u'Posterior mean', lw=2)
        ax.fill(np.concatenate([X_values, X_values[::-1]]),
                 np.concatenate([m - 1.9600 * np.sqrt(v),
                                 (m + 1.9600 * np.sqrt(v))[::-1]]),
                 alpha=.5, fc='b', ec='None', label='95% C. I.')
        ax.plot(X_values, m - 1.96 * np.sqrt(v), 'b-', alpha=0.5)
        ax.plot(X_values, m + 1.96 * np.sqrt(v), 'b-', alpha=0.5)

        ax.plot(X_train[:, index_x], y_train,
                 'r.', markersize=10, label=u'Observations')

        ax.title('Model and observations')
        ax.ylabel('f(x)')
        ax.xlabel(label_x)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax.xlim(*bounds[index_x])
        ax.grid(True)

    if len(free_dimensions) == 2:
        index_x, index_y = free_dimensions

        if not label_x:
            label_x = f'X{index_x + 1}'

        if not label_y:
            label_y = f'X{index_y + 1}'

        n = X_train.shape[0]
        points_one_color = lambda X: plt.plot(
            X[:, index_x], X[:, index_y], 'r.', markersize=10, label=u'Observations')
        X1 = np.linspace(bounds[index_x][0], bounds[index_x][1], 200)
        X2 = np.linspace(bounds[index_y][0], bounds[index_y][1], 200)
        x1, x2 = np.meshgrid(X1, X2)
        grid_point_num = 200 * 200

        X = np.zeros((grid_point_num, input_dim))

        for i, value in enumerate(fixed_values):
            if value is not None:
                X[:, i] = [value] * grid_point_num

        X[:, index_x] = x1.reshape(200 * 200)
        X[:, index_y] = x2.reshape(200 * 200)
        m, v = model.predict(X)
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.contourf(X1, X2, m.reshape(200, 200), 100)
        points_one_color(X_train)
        plt.colorbar()
        plt.ylabel(label_y)
        plt.title('Posterior mean')
        plt.axis((bounds[index_x][0], bounds[index_x][1], bounds[index_y][0], bounds[index_y][1]))
        ##
        plt.subplot(1, 2, 2)
        plt.contourf(X1, X2, np.sqrt(v.reshape(200, 200)), 100)
        plt.colorbar()
        points_one_color(X_train)
        plt.xlabel(label_x)
        plt.title('Posterior sd.')
        plt.axis((bounds[index_x][0], bounds[index_x][1], bounds[index_y][0], bounds[index_y][1]))

    if filename != None:
        ax.savefig(filename)