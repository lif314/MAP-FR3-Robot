import pyzed.sl as sl
import cv2
import numpy as np

def main():
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720 # Use HD720 opr HD1200 video mode, depending on camera type.
    init_params.camera_fps = 30  # Set FPS to 30

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Camera Open : " + repr(err) + ". Exit program.")
        exit()

    # Create a RuntimeParameters object for grabbing frames
    runtime_parameters = sl.RuntimeParameters()

    # Create sl.Mat objects to hold the images
    left_image = sl.Mat()
    right_image = sl.Mat()

    print("Press 'q' to exit the program.")

    while True:
        # Grab a new frame
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Retrieve the left and right images
            zed.retrieve_image(left_image, sl.VIEW.LEFT)
            zed.retrieve_image(right_image, sl.VIEW.RIGHT)

            # Convert the images to NumPy arrays for OpenCV
            left_frame = left_image.get_data()
            left_frame = cv2.cvtColor(left_frame, cv2.COLOR_BGRA2BGR)  # Convert from BGRA to BGR

            right_frame = right_image.get_data()
            right_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGRA2BGR)  # Convert from BGRA to BGR
   
            left_frame_resized = cv2.resize(left_frame, (640, 480))
            right_frame_resized = cv2.resize(right_frame, (640, 480))

            # Stack the left and right images horizontally
            combined_frame = np.hstack((left_frame_resized, right_frame_resized))

            # Display the combined image using OpenCV
            cv2.imshow("ZED | Left & Right Views", combined_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release resources
    cv2.destroyAllWindows()
    zed.close()

if __name__ == "__main__":
    main()
