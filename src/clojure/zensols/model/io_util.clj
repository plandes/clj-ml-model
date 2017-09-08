(ns zensols.model.io-util
  (:require [clojure.java.io :as io]
            [clojure.tools.logging :as log])
  (:require [zensols.actioncli.util :refer (trunc)])
  (:require [taoensso.nippy :as nippy]))

;; http://stackoverflow.com/questions/23018870/how-to-read-a-whole-binary-file-nippy-into-byte-array-in-clojure
(defn slurpb
  "Convert an input stream is to byte array."
  [is & {:keys [buffer-size]
         :or {buffer-size 1024}}]
  (with-open [baos (java.io.ByteArrayOutputStream.)]
    (let [ba (byte-array buffer-size)]
      (loop [n (.read is ba 0 buffer-size)]
        (when (> n 0)
          (.write baos ba 0 n)
          (recur (.read is ba 0 buffer-size))))
      (.toByteArray baos))))

(defn write-object
  "Write **obj** to **resource** using [clojure.java.io/output-stream].

  The [nippy library](https://github.com/ptaoussanis/nippy) is used for
  serialization."
  [resource obj]
  (log/infof "writing to %s, object=<%s>" resource (trunc obj))
  (with-open [out (io/output-stream resource)]
    (io/copy (nippy/freeze obj) out))
  obj)

(defn read-object
  "Write an object from **resource** using [clojure.java.io/input-stream].

  The [nippy library](https://github.com/ptaoussanis/nippy) is used for
  serialization."
  [resource]
  (log/infof "reading object from %s" resource)
  (with-open [in (io/input-stream resource)]
    (->> (slurpb in)
         nippy/thaw)))
