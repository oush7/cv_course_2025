import matplotlib.pyplot as plt
import cv2           
import numpy as np  

def solve():
    ambassadors_img_path = './data/the_ambassadors.jpg'
    output_rect_width = 200
    output_rect_height = 100

    img = cv2.imread(ambassadors_img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print("Please select 4 points on the distorted skull in this order:")
    print("1. Top-Left corner")
    print("2. Top-Right corner")
    print("3. Bottom-Right corner")
    print("4. Bottom-Left corner")
    print("Close the plot window after selecting 4 points.")

    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)
    plt.title("Select 4 points on the skull (TL, TR, BR, BL), then close")

    selected_points = plt.ginput(n=4, timeout=0, show_clicks=True)
    plt.close()

    if len(selected_points) == 4:
        src_pts = np.array(selected_points, dtype=np.float32)

        dst_pts = np.array([
            [0, 0],                           
            [output_rect_width - 1, 0],       
            [output_rect_width - 1, output_rect_height - 1],
            [0, output_rect_height - 1]   
        ], dtype=np.float32)


        perspective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

        output_size = (output_rect_width, output_rect_height)
        undistorted_skull = cv2.warpPerspective(img, perspective_matrix, output_size)
        undistorted_skull_rgb = cv2.cvtColor(undistorted_skull, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(img_rgb)
        plt.plot(src_pts[:, 0], src_pts[:, 1], 'ro-', markersize=5, linewidth=1)
        for i, (x, y) in enumerate(src_pts):
            plt.text(x + 5, y + 5, str(i + 1), color='yellow', fontsize=12)
        plt.title("Original Image with Selected Points")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(undistorted_skull_rgb)
        plt.title("Undistorted Skull")
        plt.axis('off')

        plt.tight_layout()
        plt.show()


    else:
        print("Error: Fewer than 4 points were selected.")

if __name__ == '__main__':
    solve()