known_face_encodings= []
IDs = []


# Loading video for face detection
video_capture = cv2.VideoCapture("/home/chia-yen/Downloads/Xiaomi.mp4")

frame_count = 0

while video_capture.isOpened() and frame_count <300:    
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Bail out when the video file ends
    if not ret:
        video_capture.release()
        cv2.destroyAllWindows()
        break
        
    # We will search face in every 15 frames to speed up process.
    frame_count += 1
    if frame_count % 15 == 0:    
        #rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Find all the faces and face encodings in the current frame of video
        #rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            if known_face_encodings != []:
            # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                
            # check the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            
            # Take the best one
                best_match_index = np.argmin(face_distances)
            
            # If we have a match
                if matches[best_match_index]:
                # Save the name of the best match
                    ID = IDs[best_match_index]            
                
                else:
                    ID = len(IDs)+1
                    known_face_encodings.append(face_encoding)
                    IDs.append(ID)
                    
            else:
                ID = len(IDs)+1
                known_face_encodings.append(face_encoding)
                IDs.append(ID)
        
            # draw the predicted face name on the image
            ID = str(ID)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, ID, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            
            # show the output image
        cv2.imshow("Frame", frame)
        cv2.waitKey(0)
