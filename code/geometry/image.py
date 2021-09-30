import numpy as np
import cv2
import torch
import torchvision.transforms as transforms

def skew(x):
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


def compute_fundamental_from_poses(K_src, K_dst, T_src, T_dst):
    T_src2dst = T_dst.dot(np.linalg.inv(T_src))
    R = T_src2dst[:3, :3]
    t = T_src2dst[:3, 3]
    tx = skew(t)
    E = np.dot(tx, R)
    return np.linalg.inv(K_dst).T.dot(E).dot(np.linalg.inv(K_src))


def detect_keypoints(im, detector, num_kpts=10000):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    if detector == 'sift':
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=num_kpts)
        kpts = sift.detect(gray)
    elif detector == 'orb':
        orb = cv2.ORB_create(nfeatures=num_kpts)
        kpts = orb.detect(gray)
    else:
        raise NotImplementedError('Unknown keypoint detector.')

    return kpts


def extract_feats(im, kpts, feature_type, model=None):
    if feature_type == 'sift':
        sift = cv2.xfeatures2d.SIFT_create()
        kpts, feats = sift.compute(im, kpts)

    elif feature_type == 'orb':
        orb = cv2.ORB_create()
        kpts, feats = orb.compute(im, kpts)

    elif feature_type == 'caps':
        assert model is not None
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])

        kpts = np.array([[kp.pt[0], kp.pt[1]] for kp in kpts])
        kpts = torch.from_numpy(kpts).float()

        desc_c, desc_f = model.extract_features(
            transform(im).unsqueeze(0).to(model.device),
            kpts.unsqueeze(0).to(model.device))

        feats = torch.cat((desc_c, desc_f),
                          -1).squeeze(0).detach().cpu().numpy()
    else:
        raise NotImplementedError('Unknown feature descriptor.')

    return feats


def match_feats(feats_src,
                feats_dst,
                feature_type,
                ratio_test=True,
                ratio_thr=0.6):
    if feature_type == 'orb':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        good = bf.match(feats_src, feats_dst)
    else:  # sift and caps descriptor
        if ratio_test:
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(feats_src, feats_dst, k=2)
            good = []
            for m, n in matches:
                if m.distance < ratio_thr * n.distance:
                    good.append(m)
            if len(good) < 50:
                matches = sorted(matches,
                                 key=lambda x: x[0].distance / x[1].distance)
                good = [m[0] for m in matches[:50]]

        else:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            good = bf.match(feats_src, feats_dst)
            if len(good) < 50:
                bf = cv2.BFMatcher()
                matches = bf.match(feats_src, feats_dst)
                matches = sorted(matches, key=lambda x: x.distance)
                good = [m for m in matches[:50]]

    return good


def estimate_essential(kp1, kp2, matches, K1, K2, th=1e-4):
    src_pts = np.float32([kp1[m.queryIdx].pt
                          for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt
                          for m in matches]).reshape(-1, 1, 2)
    pts_l_norm = cv2.undistortPoints(src_pts, cameraMatrix=K1, distCoeffs=None)
    pts_r_norm = cv2.undistortPoints(dst_pts, cameraMatrix=K2, distCoeffs=None)
    E, mask = cv2.findEssentialMat(pts_l_norm,
                                   pts_r_norm,
                                   focal=1.0,
                                   pp=(0., 0.),
                                   method=cv2.RANSAC,
                                   prob=0.999,
                                   threshold=th)
    if E.shape != (3, 3):
        return np.eye(3), np.zeros((len(matches))), np.eye(3), np.zeros((3))

    mask = np.squeeze(mask).astype(bool)
    _, R, t, _ = cv2.recoverPose(E, pts_l_norm[mask], pts_r_norm[mask])
    t = np.squeeze(t)
    return E, mask, R, t


def decolorize(img):
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),
                        cv2.COLOR_GRAY2RGB)


def draw_matches(kps1, kps2, tentatives, img1, img2, H, mask):
    if H is None:
        print("No homography found")
        return
    matchesMask = mask.ravel().tolist()
    h, w, ch = img1.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1],
                      [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, H)
    img2_tr = cv2.polylines(decolorize(img2), [np.int32(dst)], True,
                            (0, 0, 255), 3, cv2.LINE_AA)
    draw_params = dict(
        matchColor=(255, 255, 0),  # draw matches in yellow color
        singlePointColor=None,
        matchesMask=matchesMask,  # draw only inliers
        flags=2)
    return cv2.drawMatches(decolorize(img1), kps1, img2_tr, kps2, tentatives,
                           None, **draw_params)
