from math import sqrt

x_sample = [0,  1,  2,  3,  4]
y_sample = [1,  2,  5,  10,  17]
step_sample = 0.05
tolerance_sample = 0.01
iterations_sample = 1000

def linear_regression(x_values,  y_values,  step_size, tolerance, max_iterations):
    w0 = 0
    w1 = 0
    magnitude = float('inf')
    counter = 0
    rss_counter = []

    while magnitude >= tolerance and not(counter >= max_iterations):
        errors = [w0 + x * w1 - y for x, y in zip(x_values, y_values)]
        derivative_w0 = sum(errors)
        w0 = w0 - derivative_w0 * step_size

        y_prime = [error * x for error, x in zip(errors, x_values)]
        derivaitve_w1 = sum(y_prime)
        w1= w1 - derivaitve_w1 * step_size
       
        magnitude = sqrt(derivative_w0 ** 2 + derivaitve_w1 ** 2)

        rss = sum([x ** 2 for x in errors])

        #Print details
        print("Iteration: ", counter)
        print("errors = ", errors)
        print("magnitude = ", magnitude)
        print("w0 = ", w0, ", w1 = ", w1)
        print("rss =",rss, "\n")
        
        counter = counter + 1

    return [w0,  w1]

#Test it out
weights = linear_regression(x_sample, y_sample, step_sample, tolerance_sample, iterations_sample)
print("Final weights = ",  weights)

