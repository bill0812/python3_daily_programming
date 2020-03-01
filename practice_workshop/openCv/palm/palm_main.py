import crop_palm , os , cv2

if __name__ == "__main__" :
    input_dir = "01_palm/"
    output_dir = "01_palm_crop/"

    if not os.path.exists(output_dir) :
        os.makedirs(output_dir)

    for img in os.listdir(input_dir):
        if img.endswith(".png"):
            print(os.path.join(input_dir, img)[8:-4])
            # crop_palm.deal_img(os.path.join(input_dir, img) , output_dir)