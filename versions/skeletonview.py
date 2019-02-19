class SkeletonDataset:
    def __init__(self, a_=6, s_=10, e_=1):
        self.path = "C:\\Users\\gcper\\Code\\STEM\\data\\MSRDailyAct3D_pack1\\"
        self.template = "a{}_s{}_e{}_skeleton.txt"
        self.skeletons = []
        self.boxes = []
        s = s_ - 1
        for a in range(a_):  # out of 2
            if True:  # out of 10
                for e in range(e_):  # out of 2
                    # print("%.1f%% skeletons loaded" % (((a / a_) + (s / (a_ * s_)) + (e / (a_ * s_ * e_))) * 100))
                    filename = self.template.format(*[str(i + 1).zfill(2) for i in [a, s, 1]])
                    print(filename)
                    skeleton_file = SkeletonFile(self.path + filename)
                    self.skeletons.extend(skeleton_file.skeletons)
                    self.boxes.extend(skeleton_file.box)


class SkeletonFile:
    def __init__(self, path):
        self.path = path
        self.skeletons = []
        self.frames = 0
        self.joints = 0
        self.box = []
        self.read_skeletons()

    def read_skeletons(self):
        with open(self.path, 'r') as f:
            data = iter([i.replace('\n', '') for i in f.readlines()])
            self.frames, self.joints = map(int, next(data).split())
            for frame in range(self.frames):
                joints = int(next(data))
                if joints == 80:
                    print(self.path)
                    import sys
                    sys.exit()
                else:
                    self.skeletons.append([])
                    for joint in range(joints // 2):
                        x, y, z, _ = map(float, next(data).split())
                        u, v, depth, _ = map(float, next(data).split())
                        self.skeletons[-1].append([u * 320, v * 240])
                self.make_box(self.skeletons[-1])

    def make_box(self, skeleton):
        xs, ys = zip(*skeleton)
        mx, my, nx, ny = [min(xs), min(ys), max(xs), max(ys)]
        out = [[min(xs), min(ys)], [max(xs), max(ys)]]

        out[0][0] -= (nx - mx) * .1
        out[0][0] = max(0, out[0][0])
        out[1][0] += (nx - mx) * .1
        out[1][0] = min(320, out[1][0])
        out[0][1] -= (ny - my) * .1
        out[0][1] = max(0, out[0][1])
        out[1][1] += (ny - my) * .1
        out[1][1] = min(240, out[1][1])
        self.box.append(out)
