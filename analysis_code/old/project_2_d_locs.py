import pandas as pd


import numpy as np
import matplotlib.pyplot as plt


def cartesian_to_spherical(x, y, z):
    """
    Convert Cartesian coordinates to spherical coordinates.

    Parameters:
        x, y, z: Cartesian coordinates.

    Returns:
        r: Radius.
        theta: Azimuthal angle (longitude) in radians.
        phi: Polar angle (colatitude) in radians.
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)  # Longitude
    phi = np.arccos(z / r)  # Colatitude
    return r, theta, phi


def azimuthal_equidistant_projection(r, theta, phi, theta_0=0, phi_0=0):
    """
    Apply the Azimuthal Equidistant Projection to spherical coordinates.

    Parameters:
        r: Radius.
        theta: Azimuthal angle (longitude) in radians.
        phi: Polar angle (colatitude) in radians.
        theta_0: Central longitude (default=0).
        phi_0: Central colatitude (default=0).

    Returns:
        x_prime: Projected x-coordinate.
        y_prime: Projected y-coordinate.
    """
    # Convert colatitude to latitude
    lat = (np.pi / 2) - phi
    lat_0 = (np.pi / 2) - phi_0

    # Angular distance c
    cos_c = np.sin(lat_0) * np.sin(lat) + np.cos(lat_0) * np.cos(lat) * np.cos(
        theta - theta_0
    )
    cos_c = np.clip(cos_c, -1.0, 1.0)  # Ensure the value is within valid range
    c = np.arccos(cos_c)

    # Handle division by zero for c = 0
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.where(c == 0, 1, c / np.sin(c))

    # Projected coordinates
    x_prime = r * k * np.cos(lat) * np.sin(theta - theta_0)
    y_prime = (
        r
        * k
        * (
            np.cos(lat_0) * np.sin(lat)
            - np.sin(lat_0) * np.cos(lat) * np.cos(theta - theta_0)
        )
    )

    return x_prime, y_prime


# Example usage:
if __name__ == "__main__":
    # Sample 3D coordinates (replace with your actual data)

    channel_positions = pd.read_csv(
        "Co-registered average positions.pos",
        header=None,
        delimiter="\t",
        names=["electrode", "x", "y", "z"],
    )

    coordinates_3d = channel_positions[["x", "y", "z"]].values

    # Initialize lists to store spherical coordinates and projected coordinates
    r_list = []
    theta_list = []
    phi_list = []

    # Convert each Cartesian coordinate to spherical coordinates
    for x, y, z in coordinates_3d:
        r, theta, phi = cartesian_to_spherical(x, y, z)
        r_list.append(r)
        theta_list.append(theta)
        phi_list.append(phi)

    r_array = np.array(r_list)
    theta_array = np.array(theta_list)
    phi_array = np.array(phi_list)

    # Apply the Azimuthal Equidistant Projection
    # Central point can be adjusted by changing theta_0 and phi_0
    x_prime, y_prime = azimuthal_equidistant_projection(
        r_array, theta_array, phi_array, theta_0=0, phi_0=0
    )

    # Plot the projected points
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(-x_prime, -y_prime, color="blue")

    # Annotate each point
    for i, x_p, y_p in zip(channel_positions["electrode"].values, -x_prime, -y_prime):
        plt.text(x_p, y_p, f"{i}", fontsize=12, ha="right", va="bottom")

    plt.title("Azimuthal Equidistant Projection of 3D Head Coordinates")
    plt.xlabel("x'")
    plt.ylabel("y'")
    plt.grid(True)
    plt.axis("equal")
    fig.savefig("locs.svg")


    channel_positions_new = pd.DataFrame({'electrodes': channel_positions["electrode"].values, 'x': -x_prime, 'y': -y_prime})
    channel_positions_new.to_csv("pos_2_d.csv")

