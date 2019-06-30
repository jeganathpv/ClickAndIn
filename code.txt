import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import pandas as pd
import cv2
from datetime import datetime
# import matplotlib.pyplot as plt
import numpy as np
# import matplotlib.cbook


import mysql.connector as pymysql
connection = pymysql.connect(host='localhost',user='newuser',passwd='password',database='click')
cursor=connection.cursor()




ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
test_path = "training_set/test"

def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    X = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(
                        face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(
                    image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(
        n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception(
            "Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []
    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(
        X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <=
                   distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


def show_prediction_labels_on_image(img_path, predictions):
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode("UTF-8")

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10),
                        (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5),
                  name, fill=(255, 255, 255, 255))

    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    pil_image.show()


if __name__ == "__main__":

    df=pd.read_csv('AttendanceReport.csv')

    def export_csv(presentees, df):
        time=datetime.now().strftime("%d/%m/%Y-%H:%M")
        attendance=[]
        for row in df.itertuples():
            if row.Names in presentees:
                attendance.append(1)
                sql="""INSERT INTO attendance(name, date, attend) values ('{}','{}',{})""".format(row.Names,time,1)
                
                cursor.execute(sql)
            else:
                attendance.append(0)
                sql="""INSERT INTO attendance(name, date, attend) values ('{}','{}',{})""".format(row.Names,time,0)
                
                cursor.execute(sql)
        df[time]=attendance
        dimension=df.shape
        percentage=[]
        totalclasses=float(dimension[1]-2)
        for row in df.itertuples():
            count=0
            for val in row[3:]:
                if val==1:
                    count+=1
            per=round(float(count/totalclasses)*100.0,2)
            # print(count,totalclasses,per)
            percentage.append(per)
        df['Percentage']=percentage
        df.to_csv('AttendanceReport.csv', sep=',',index=False)

    # def analysis(df):
    #     plt.rcdefaults()
    #     fig, ax = plt.subplots()
    #     # Example data
    #     people = tuple(df.Names)
    #     y_pos = np.arange(len(people))
    #     performance = tuple(df.Percentage)
    #     error = np.random.rand(len(people))
    #     ax.barh(y_pos, performance, xerr=error, align='center',color='blue')
    #     ax.set_yticks(y_pos)
    #     ax.set_yticklabels(people)
    #     ax.invert_yaxis()  # labels read top-to-bottom
    #     ax.set_xlabel('Percentage')
    #     ax.set_title('Attentance Percentage')
    #     plt.show()


    def cam_data():
        cam = cv2.VideoCapture(0)
        cv2.namedWindow("test")
        img_counter = 0
        while True:
            ret, frame = cam.read()
            cv2.imshow("test", frame)
            if not ret:
                break
            k = cv2.waitKey(1)

            if k % 256 == 32:
                # SPACE pressed
                img_name = "opencv_frame_{}.png".format(img_counter)
                cv2.imwrite('./camera/' + img_name, frame)
                print("{} written!".format(img_name))
                img_counter += 1
                break
        cam.release()
        cv2.destroyAllWindows()
        testset_code('./camera')


    def trainset_code():
        print("Training KNN classifier...")
        classifier = train(
            "training_set/train", model_save_path="trained_knn_model.clf", n_neighbors=2)
        print("Training complete! \n")

    def testset_code(test_path):
        for image_file in os.listdir(test_path):
            full_file_path = os.path.join(test_path, image_file)
            print("Looking for faces in {}".format(image_file))
            # Find all people in the image using a trained classifier model
            # Note: You can pass in either a classifier file name or a classifier model instance
            predictions = predict(
                full_file_path, model_path="trained_knn_model.clf")
            presentees=[]
            # Print results on the console
            for name, (top, right, bottom, left) in predictions:
                print("- Found {} at ({}, {})".format(name, left, top))
                presentees.append(name)
            show_prediction_labels_on_image(os.path.join(
                test_path, image_file), predictions)
        export_csv(presentees, df)
        print('Attendance Updated')
        connection.commit()
        connection.close()

    dec = int(input('1. Train and Test\n2. Train\n3. Test\n4. Camera Data\n\nEnter Your Choice: '))
    if(dec == 1):
        trainset_code()
        testset_code(test_path)
    elif(dec == 2):
        trainset_code()
    elif(dec == 3):
        testset_code(test_path)
    elif(dec == 4):
        cam_data()
    elif(dec == 5):
        analysis(df)