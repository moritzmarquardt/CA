import darsia as da

# Create a darsia Image: An image that also contains information of physical entities
image = da.Image("c:/Users/marqu/thesis/CA/images/baseline.jpg", origin = [5, 2], width = 280, height = 150)

# Use the show method to take a look at the imported image (push any button to close the window)
image.show()

# Copies the image and adds a grid on top of it.
grid_image = image.add_grid(origin = [5, 2], dx = 10, dy = 10)
grid_image.show()

# Extract region of interest (ROI) from image:
ROI_image = da.extractROI(image, [[150, 0], [280, 70]])
ROI_image.show()