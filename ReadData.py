from DataUtils import *
import threading, cv2, sys, time

class ReadData():
    def __init__(self, dsName='airsim', subType='mr', seq=0):
        self.dsName = dsName
        self.subType = subType
        self.path = getPath(dsName, seq=seq, subType=subType)

        if dsName == 'airsim':
            self.data = pd.read_csv(self.path + 'data.txt', sep=' ', header=None)
            self.time_stamp = self.data.iloc[:, 0].values
        else:
            self.time_stamp = None

        # images
        self.imgNames = getImgNames(self.path, dsName, ts=self.time_stamp, subType=subType)
        print(self.imgNames)
        self.numImgs = len(self.imgNames)
        self.numChannel = 3 if self.dsName is not 'euroc' else 1
        self.imgs = np.zeros((self.numImgs, self.numChannel, 360, 720), dtype=np.float32)
        self.getImages()

    def getNewImgNames(self, subtype='bar'):
        return getImgNames(self.path, self.dsName, self.time_stamp, subType=subtype)

    def getImgsFromTo(self, start, N):
        if start > self.numImgs:
            sys.exit('ReadData-getImgsFromTo: this should be the case')

        end, N = getEnd(start, N, self.numImgs)
        print('PrepData-reading imgs from %d to %d(): reading imgs' % (start, end))
        for i in range(start, end):
            fName = self.imgNames[i]
            if self.dsName == 'euroc':
                img = cv2.imread(fName, 0) / 255.0
            else:
                img = cv2.imread(fName) / 255.0
            if self.dsName is not 'airsim':
                img = cv2.resize(img, (720, 360))
            img = np.reshape(img.astype(np.float32), (-1, self.numChannel, 360, 720))
            self.imgs[i, :] = img  # no lock is necessary
        print('PrepData-reading imgs from %d to %d(): done reading imgs' % (start, end))

    def getImages(self):
        partN = 500
        nThread = int(self.numImgs / partN) + 1
        print('# of thread reading imgs: %d' % (nThread))
        threads = []
        for i in range(0, nThread):
            start = i * partN
            threads.append(threading.Thread(target=self.getImgsFromTo, args=(start, partN)))
            threads[i].start()

        for thread in threads:
            thread.join()  # wait until this thread ends ~ bit of loss in time..


if __name__ == '__main__':
    ReadData(dsName='airsim', subType='mr', seq=2)