import csv
import pickle
from pathlib import Path

from tqdm import tqdm
import numpy as np
import face_recognition
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from PIL import Image, ImageDraw


def list_categories(image_base_dir):
    image_base_path = Path(image_base_dir)
    image_path_list = [p for p in image_base_path.iterdir() if p.is_dir()]
    categories = [p.name for p in image_path_list]
    return categories


def find_all_image_path(image_base_dir):
    image_base_path = Path(image_base_dir)
    globstr = '*/*'
    allow_suffix = ['.jpg', '.jpeg', '.png', '.svg']
    _all_image_path = list(image_base_path.glob(globstr))
    all_image_path = []
    for path in _all_image_path:
        if not path.is_file():
            continue
        if path.suffix not in allow_suffix:
            continue
        all_image_path.append(path)
    return all_image_path


def make_model(pathlist, categories):
    x = make_facenet_matrics(pathlist)
    y = make_answer_matrics(pathlist, categories)
    svc, score = _make_model_SVC(x, y)
    return svc, score


def make_facenet_matrics(imagepathlist, feature_num=128):
    cnv_list = None
    for imagepath in tqdm(imagepathlist):
        cnv = load_face_image(imagepath, feature_num)
        if cnv_list is None:
            cnv_list = cnv[0]
        else:
            cnv_list = np.concatenate([cnv_list, cnv[0]])
    reshape_cnv_list = cnv_list.reshape(len(imagepathlist),
                                        feature_num)
    return reshape_cnv_list


def load_face_image(imagepath, feature_num=128):
    image = load_image(imagepath)
    face_location = (0, image.shape[1], image.shape[0], 0)
    cnv = face_recognition.face_encodings(image,
                                          known_face_locations=[face_location])
    return cnv


def make_answer_matrics(pathlist, categories):
    ans_list = []
    for path in pathlist:
        dirname = path.parent.name
        index = categories.index(dirname)
        ans_list.append(index)
    return ans_list


def _make_model_SVC(x, y):
    train_test = train_test_split(x, y, random_state=0, train_size=0.8)
    xtrain, xtest, ytrain, ytest = train_test
    svc = SVC(probability=True)
    svc.fit(xtrain, ytrain)
    ypred = svc.predict(xtest)
    score = accuracy_score(ytest, ypred) * 100.0
    return svc, score


def save_model(svc, output):
    outpath = Path(output)
    parent = outpath.parent
    if not parent.exists():
        parent.mkdir(parents=True)
    with outpath.open('wb') as fd:
        pickle.dump(svc, fd)


def load_model(modelfile):
    modelpath = Path(modelfile)
    model = None
    with modelpath.open('rb') as fd:
        model = pickle.load(fd)
    return model


def save_categories(categories, output):
    outpath = Path(output)
    with outpath.open('w') as fd:
        writer = csv.writer(fd)
        writer.writerow(categories)


def load_categories(csvfile):
    csvpath = Path(csvfile)
    memberlist = []
    with csvpath.open('r') as fd:
        reader = csv.reader(fd)
        memberlist = next(reader)
    return memberlist


def load_image(imagepath):
    return face_recognition.load_image_file(imagepath)


def find_face_locations(image):
    face_locations = face_recognition.face_locations(image)
    return face_locations


def predict_all_face(svc, face_locations, image):
    cnv = face_recognition.face_encodings(image,
                                          known_face_locations=face_locations)
    result = svc.predict_proba(cnv)
    return result


def save_result_image(image, output, face_locations, results, categories):
    pil_image = Image.fromarray(image)
    pil_image_draw = ImageDraw.Draw(pil_image)
    for face_location, result in zip(face_locations, results):
        rate, member = max(zip(result, categories))
        location = _convert_pil_location(face_location)
        pil_image_draw.rectangle(location)
        label = f'{member}[{round(rate*100)}%]'
        pil_image_draw.text(location[:2], label)
    pil_image.save(output)


def _convert_pil_location(face_location):
    x1, y1, x2, y2 = face_location
    xstart = min(x1, x2)
    xend = max(x1, x2)
    ystart = min(y1, y2)
    yend = max(y1, y2)
    return (ystart, xstart, yend, xend)
