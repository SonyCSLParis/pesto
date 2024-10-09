import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# Sample data generator (infinite loop)
def data_generator():
    iter_num = 0
    while True:  # Infinite loop
        f0 = random.uniform(100, 1000)
        vol = random.uniform(0.1, 1.0)
        yield iter_num, f0, vol
        iter_num += 1


if __name__ == '__main__':
    # Set up the plot
    fig, ax = plt.subplots()
    scat = ax.scatter([], [])

    # Initialize the plot limits
    ax.set_xlim(0, 100)  # Adjust as needed for your iterations
    ax.set_ylim(0, 1000)  # Adjust as needed for f0 range

    # Function to update the scatter plot
    def update(data):
        iter_num, f0, vol = data
        scat.set_offsets([iter_num, f0])
        scat.set_sizes([vol * 100])  # Scale vol for point size

    # Animate the plot
    ani = animation.FuncAnimation(fig, update, frames=data_generator, interval=100, cache_frame_data=False, blit=False)

    # Show the plot
    plt.show()
