import numpy as np

def argmax(out):
    return np.argmax(out)

def classification_output(out, batch=1):
    batch_len = len(out) // batch
    assert batch_len * batch == len(out)
    for i in range(0, len(out), batch_len):
        tmp = out[i*batch_len:(i+1)*batch_len]

        print("\n*** Batch=%s ***" % i)
        print("Front 10 numbers: [%s]" % \
              " ".join([str(d) for d in tmp[:10]]))
        print("Last  10 numbers: [%s]" % \
              " ".join([str(d) for d in tmp[-10:]]))
        print("Argmax output category: %d" % argmax(tmp))

def detection_output(out, batch=1):
    batch_len = len(out) // batch
    assert batch_len * batch == len(out)
    for i in range(0, len(out), batch_len):
        tmp = out[i*batch_len:(i+1)*batch_len]

        print("\n*** Batch=%s ***" % i)
        for i in range(0, len(tmp), 6):
           if tmp[i] == -1:
               print ("Detect object number: %d" % (i // 6))
               break
           print (tmp[i:i+6])
