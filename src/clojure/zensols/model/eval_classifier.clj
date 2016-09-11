(ns ^{:doc
"A *client* entry point library to help with evaluating machine learning
models.  This library not only wraps the Weka library but also provides
additional functionality like a two pass cross
validation (see [[*two-pass-config*]])."
      :author "Paul Landes"}
    zensols.model.eval-classifier
  (:use [clojure.java.io :as io :only [input-stream output-stream file]])
  (:use [clojure.pprint :only [pprint]])
  (:require [clojure.tools.logging :as log]
            [clojure.string :as str]
            [clojure.stacktrace :refer (print-stack-trace)]
            [clojure.data.csv :as csv])
  (:require [zensols.tabres.display-results :as dr])
  (:require [zensols.model.execute-classifier :refer (model-config) :as exc]
            [zensols.model.weka :as weka]
            [zensols.model.classifier :as cl]))

(def ^:dynamic *cross-fold-count*
  "The default number of folds to use during cross fold
  validation (see [[cmpile-results]])."
  10)

(def ^:dynamic *default-set-type*
  "The default type of test, which is one of:

      * `:cross-validation`: run a N fold cross validation (default)
      * `:test-train`: train the classifier and then evaluate"
  :cross-validation)

(defn print-model-config
  "Pretty print the model configuation set with [[with-model-conf]]."
  []
  (pprint (model-config)))

(defn read-arff
  "Read the ARFF file configured with [[with-model-conf]]."
  []
  (let [model-conf (model-config)
        file-name (format "%s-data.arff" (:name model-conf))]
    (cl/read-arff (io/file (cl/analysis-report-resource) file-name))))

(defn write-arff
  "Write the ARFF file configured with [[with-model-conf]]."
  []
  (let [model-conf (model-config)
        file-name (format "%s-data.arff" (:name model-conf))
        file (io/file (cl/analysis-report-resource) file-name)]
    (binding [cl/*arff-file* file]
      (cl/write-arff (exc/cross-fold-instances)))
    file))

(defn display-features
  "Display features as configured in a model with [[with-model-conf]].

  For the non-zero-arg form, see [[with-model-conf]]."
  ([]
   (let [{:keys [feature-metas-fn class-feature-meta-fn create-feature-sets-fn]}
         (model-config)]
     (display-features (feature-metas-fn)
                       (class-feature-meta-fn)
                       (create-feature-sets-fn))))
  ([feature-metas class-feature-meta feature-sets]
   (let [keys (map first (concat (list class-feature-meta) feature-metas))]
     (->> feature-sets
          (map (fn [tok]
                 (map (fn [tkey]
                        (get tok tkey))
                      keys)))
          ((fn [data]
             (dr/display-results data
                                 :column-names (map name keys))))))))

(defn- cross-validate-results-struct
  "Create a data structure with cross validation results."
  [classifiers attribs-sets]
  (log/debugf "cvr struct classifiers=<%s>, attribs=%s"
              (pr-str classifiers) (pr-str attribs-sets))
  (letfn [(cr-test [classifier attribs]
            (try
              (log/infof "classifier: %s, %s" classifier (pr-str attribs))
              (let [res (cl/cross-validate-tests classifier (map name attribs))]
                (log/debug (with-out-str (cl/print-results res)))
                res)
              (catch Exception e
                (let [msg (format "Can't cross validate classifier %s, attrib: %s: message=%s"
                                  classifier (if attribs (str/join "," attribs))
                                  (.toString e))]
                  (log/error (str "stack trace: " (print-stack-trace e)))
                  (log/error e msg)
                  nil))))]
    (->> classifiers
         (map (fn [classifier]
                (filter identity
                        (map #(cr-test classifier %)
                             attribs-sets))))
         (filter #(not (empty? %)))
         (apply concat))))

(defn- cross-validate-results
  "Cross validate several models in series.

  * **classifier-sets** is a key in [[zensols.model.weka/*classifier*]] or a
  constructed classifier (see [[zensols.model.weka/make-classifiers]])

  * **feature-sets-key** identifies what feature set (see
  **:feature-sets-set** in [[with-model-conf]])"
  [classifier-sets feature-sets-key]
  (log/debugf "cvr struct attrib key=%s" feature-sets-key)
  (let [model-conf (model-config)]
    (log/debugf "model config keys: %s" (keys model-conf))
    (let [classifier-attrib (first (exc/model-classifier-label))
          feature-meta-sets (get (:feature-sets-set model-conf)
                               feature-sets-key)
          _ (assert feature-meta-sets (format "no such feature meta set: <%s>"
                                            feature-sets-key))
          instances (exc/cross-fold-instances)
          num-inst (.numInstances instances)
          folds *cross-fold-count*]
      (log/debugf "number of instances: %d, feature-metas: <%s>"
                  num-inst (pr-str feature-meta-sets))
      (log/tracef "instances class: %s" (-> instances .getClass .getName))
      (if (<= num-inst folds)
        (throw (ex-info "Not enough folds"
                        {:num-inst num-inst
                         :folds folds}))
        (binding [cl/*get-data-fn* #(identity instances)
                  cl/*class-feature-meta* (name classifier-attrib)
                  cl/*cross-fold-count* folds]
          (->> classifier-sets
               (map #(cross-validate-results-struct
                      (weka/make-classifiers %) feature-meta-sets))
               (apply concat)
               doall))))))

(defn test-train-results
  "Test the performance of a model by training on a given set of data
  and evaluate on the test data.

  See [[train-model]] for parameter details."
  [classifier-sets feature-sets-key]
  (log/debugf "feature-sets-key=%s, classifier-sets=%s"
              feature-sets-key classifier-sets)
  (let [model-conf (model-config)
        _ (log/debugf "model config keys: %s" (keys model-conf))
        classifier-attrib (first (exc/model-classifier-label))
        feature-meta-sets (get (:feature-sets-set model-conf)
                               feature-sets-key)
        _ (log/debugf "feature meta sets: <%s>"
                      (pr-str feature-meta-sets))
        {train-instances :train test-instances :test}
        (exc/train-test-instances)]
    (assert feature-meta-sets (format "no such feature meta set: <%s>"
                                      feature-sets-key))
    (assert train-instances "train-instances")
    (assert test-instances "test-instances")
    (log/debugf "number of train/test instances:(%d, %d), feature-metas: <%s>"
                (.numInstances train-instances)
                (.numInstances test-instances)
                (pr-str feature-meta-sets))
    (log/tracef "train instances class: %s"
                (-> train-instances .getClass .getName))
    (log/tracef "test instances class: %s"
                (-> test-instances .getClass .getName))
    (binding [cl/*class-feature-meta* (name classifier-attrib)]
      (->> classifier-sets
           (map weka/make-classifiers)
           (apply concat)
           (map #(cl/train-test-classifier % feature-meta-sets
                                           train-instances test-instances))
           (apply concat)
           doall))))

(defn compile-results
  "Run cross-fold validation and compile into a nice results map sorted by
  performance.

  See [[zensols.model.classifier/compile-results]].

  * **classifier-sets** is a key in [[zensols.model.weka/*classifiers*]] or a
  constructed classifier (see [[zensols.model.weka/make-classifiers]])

  * **feature-sets-key** identifies what feature set (see
  **:feature-sets-set** in [[with-model-conf]])

  Keys
  ----
  * **:test-type** the type of test to run (see [[*default-set-type*]])"
  [classifier-sets feature-set-key &
   {:keys [test-type] :or {test-type *default-set-type*}}]
  (let [test-fn (case test-type
                  :cross-validation cross-validate-results
                  :test-train test-train-results)
        res (test-fn classifier-sets feature-set-key)]
    (log/debugf "compile results count %d" (count res))
    (cl/compile-results res)))

(defn terse-results
  "Return terse cross-validation results in an array:
  * classifier name
  * weighted F-measure
  * feature-metas

  Parameters
  ----------
  * **classifier-sets** is a key in [[zensols.model.weka/*classifiers*]] or a
  constructed classifier (see [[zensols.model.weka/make-classifiers]])

  * **feature-sets-key** identifies what feature set (see
  **:feature-sets-set** in [[with-model-conf]])

  Keys
  ----
  * **:only-stats?** if `true` only return statistic data
  * **:test-type** the type of test to run (see [[*default-set-type*]])"
  [classifier-sets feature-set-key &
   {:keys [only-stats? test-type]
    :or {only-stats? true
         test-type *default-set-type*}}]
  (let [res (compile-results classifier-sets feature-set-key)]
    (log/debugf "terse results count %d" (count res))
    (map (fn [elt]
           (concat [(cl/classifier-name (:classifier elt))
                    (:wfmeasure elt)]
                   (if-not only-stats?
                     [feature-set-key
                      (:feature-metas elt)])))
         res)))

(defn print-best-results
  "Print the highest (best) scored cross validation information.

  * **classifier-sets** is a key in [[zensols.model.weka/*classifiers*]] or a
  constructed classifier (see [[zensols.model.weka/make-classifiers]])

  * **feature-sets-key** identifies what feature set (see
  **:feature-sets-set** in [[with-model-conf]])"
  [classifier-sets feature-set-key]
  (let [comp-res (compile-results classifier-sets feature-set-key)]
    (cl/print-results (:result (first comp-res)))))

(defn create-model
  "Create a model that can be trained.  This runs cross fold validations to
  find the best classifier and feature set into a result that can be used
  with [[train-model]] and subsequently [[write-model]].

  * **classifier-sets** is a key in [[zensols.model.weka/*classifiers*]] or a
  constructed classifier (see [[zensols.model.weka/make-classifiers]])

  * **feature-sets-key** identifies what feature set (see
  **:feature-sets-set** in [[with-model-conf]])"
  [classifier-sets feature-set-key]
  (-> (compile-results classifier-sets feature-set-key)
      first
      (dissoc :result :all-results)
      (merge {:name (:name (model-config))
              :create-time (java.util.Date.)})))

(defn train-model
  "Train a model created from [[create-model]].  The model is trained on the
  full available dataset.  After the classifier is trained, you can save it to
  disk by calling [[write-model]].

  * **model** a model that was created with [[create-model]]"
  [model]
  (let [classifier (:classifier model)
        attribs (map name (:feature-metas model))
        classify-attrib (first (exc/model-classifier-label))
        instances (exc/cross-fold-instances)]
    (binding [cl/*get-data-fn* #(identity instances)
              cl/*class-feature-meta* (name classify-attrib)]
      (log/infof "training model %s classifier %s with %d instances"
                 (:name (model-config))
                 (-> classifier .getClass .getName)
                 (.numInstances instances))
      (cl/train-classifier classifier attribs)
      model)))

(defn- model-persist-name []
  (:name (model-config)))

(defn write-model
  "Persist/write the model to disk.

  * **model** a model that was trained with [[train-model]]

  See [[zensols.model.classifier/model-dir]] for information about to where the
  model is written."
  [model]
  (let [model-conf (model-config)
        context-fn (:context-fn model-conf)
        persist-name (model-persist-name)]
    (log/tracef "saving classifer %s as %s"
                (type (:classifier model)) persist-name)
    (let [context (if context-fn (context-fn))
          model (if context
                  (assoc model :context context)
                  model)]
      (cl/write-model persist-name model))
    model))

(defn read-model
  "Read a model that was previously persisted to the file system.

  See [[zensols.model.classifier/model-dir]] for where the model is read from."
  []
  (cl/read-model (model-persist-name)))

(defn eval-and-write
  "Perform a cross validation and write the results to an Excel formatted file.

  See [[zensols.model.classifier/analysis-report-resource]] for where the file is
  written.

  * **classifier-sets** is a key in [[zensols.model.weka/*classifiers*]] or a
  constructed classifier (see [[zensols.model.weka/make-classifiers]])

  * **feature-sets-key** identifies what feature set (see
  **:feature-sets-set** in [[with-model-conf]])"
  [classifier-sets set-key]
  (letfn [(output-file [name]
            (let [model-conf (model-config)]
              (io/file (cl/analysis-report-resource)
                       (format "%s-%s.xls" (:name model-conf) name))))]
    (let [output-file (output-file "classification")
          model-conf (model-config)]
      (cl/excel-results
       [{:sheet-name (format "%s Classification" (str/capitalize (:name model-conf)))
         :results (cross-validate-results classifier-sets set-key)}]
       output-file)
      (log/infof "wrote results file: %s" output-file)
      output-file)))

(defn train-test-series
  "Test and train with different rations and return the results.  The return
  data writable directly as an Excel file.  However, you can also save it as a
  CSV with [[write-csv-train-test-series]].

  The keys are the classifier name and the values are the 2D result matrix."
  [classifiers meta-set divide-ratio-config]
  (let [{:keys [start stop step]} divide-ratio-config
        {:keys [divide-by-set]} (model-config)]
    (->> classifiers
         (map (fn [classifier]
                (->> (range (* start 100) (* stop 100) (* step 100))
                     (map #(/ % 100))
                     (map (fn [divide-ratio]
                            (log/infof "dividing train/test split by %.3f"
                                       divide-ratio)
                            (divide-by-set divide-ratio)
                            (compile-results (list classifier) meta-set :test-type :test-train)))
                     (apply concat)
                     (map (fn [res]
                            (let [name (-> res :classifier cl/classifier-name)]
                              {:name name
                               :stats (map #(get res %) [:train-total :test-total :wfmeasure :wprecision :wrecall])})))
                     (#(hash-map (-> % first :name)
                                 (->> (map :stats %)
                                      (cons ["Train" "Test" "F-Measure" "Precision" "Recall"])))))))
         (apply merge))))

(defn write-csv-train-test-series
  "Write the results produced with [[train-test-series]] as a CSV file to the
  analysis directory."
  [res]
  (let [{model-name :name} (model-config)]
    (->> res
         (map (fn [[classifier-name data]]
                (let [out-file (io/file (cl/analysis-report-resource)
                                        (format "%s-%s-train-test-series.csv"
                                                model-name classifier-name))]
                  (with-open [writer (io/writer out-file)]
                    (csv/write-csv writer data)))))
         doall)))


;; two pass
(def ^:dynamic *two-pass-config*
  "When this is non-`nil` use two fold cross validation.

  Description
  -----------

  Two pass validation is a term used in this library.  It means during
  cross-validation the entire data set is evaluated and (usually) statistics or
  some other additional modeling happens.

  Take for example you want to count words (think Naive Bays spam filter).  If
  create features for the entire dataset before cross-validation you're
  *cheating* because the features are based on data not seen from the test
  folds.

  To get more accurate performance metrics you can provide functions that takes
  the current training fold, compute your word counts and create your features.
  During the testing phase, the computed data is provided to create features
  based on only that (current) fold.

  To use two pass validation ever feature set needs a unique key (not needed as
  a feature).  This key is then given to a function during validation to get
  the corresponding feature set that is to be *stitched* in.

  Keys
  ----

  This variable is a map with the following keys:

  * **:id-key** a function that takes a key as input and returns a feature set

  * **:feature-metas-fn the same function as described
  in [[zensols.model.execute-classifier/with-model-conf]]

  * **:create-features-fn** the same function as descrbed
  in [[zensols.model.execute-classifier/with-model-conf]]

  * **:context-fn** the same function as descrbed
  in [[zensols.model.execute-classifier/with-model-conf]]

  *Note*: When this variable is
  bound, [[zensols.model.classifier/*cross-val-fns*]] needs to be bound as well
  to a map that uses the `two-pass-*` functions.

  See [[zensols.model.classifier/*cross-val-fns*]]"
  nil)

(defn two-pass-model
  "Create a two pass model, which should be merged with the model created
  with [[zensols.model.execute-classifier/with-model-conf]].

  * **model** the model created per documentation
  at [[zensols.model.execute-classifier/with-model-conf]]

  * **id-key** the `:id-key` documented in [[*two-pass-config*]]"
  [model id-key]
  (let [id-key (-> id-key name symbol)]
    (->> model
         :feature-sets-set
         (map (fn [[k v]]
                {k (map #(cons id-key %) v)}))
         (apply merge)
         (hash-map :feature-sets-set)
         (merge model))))

(defn two-pass-train-instances [insts state org anon-by-id-fn & rest]
  (log/debugf "training set: %d, org=%d"
              (.numInstances insts)
              (.numInstances org))
  (let [id-key (:id-key *two-pass-config*)
        _ (assert id-key "anon-id-key")
        id-key (name id-key)
        id-attrib (weka/attribute-by-name insts id-key)
        _ (assert id-attrib (format "weka attribute for <%s>" id-key))
        anon-ids (map (fn [row]
                        (int (-> insts
                                 (.instance row)
                                 (.value id-attrib))))
                      (range (.numInstances insts)))
        _ (log/infof "training %d instances" (count anon-ids))
        anons (map anon-by-id-fn anon-ids)
        _ (log/debugf "invoking context fn: %s" (:context-fn *two-pass-config*))
        context ((:context-fn *two-pass-config*) anons)
        _ (log/debugf "invoking metadata fn: %s" (:feature-metas-fn *two-pass-config*))
        feature-metas ((:feature-metas-fn *two-pass-config*) context)
        create-features-fn (:create-features-fn *two-pass-config*)]
    (swap! state assoc
           :context context
           :feature-metas feature-metas
           :two-pass-config *two-pass-config*)
    (letfn [(instance-val-fn [inst]
              (log/tracef "value func for instance: <%s>" inst)
              (let [feats (->> (.value inst id-attrib) int anon-by-id-fn :parse-anon
                               (create-features-fn context))]
                (log/tracef "returning feats: %s" feats)
                feats))]
      (log/debugf "attributes: %s" (pr-str feature-metas))
      (log/debugf "invoking weka layer with key: %s" id-key)
      (-> insts
          (weka/map-merge-instances feature-metas instance-val-fn)
          (weka/remove-attributes [id-key])))))

(defn two-pass-test-instances [insts train-state anon-by-id-fn & rest]
  (log/debugf "testing set: %d" (.numInstances insts))
  (let [{:keys [context feature-metas two-pass-config]} @train-state
        {:keys [id-key create-features-fn]} two-pass-config
        id-key (name id-key)
        id-attrib (weka/attribute-by-name insts id-key)]
    (log/debugf "id-key: %s" id-key)
    (letfn [(instance-val-fn [inst]
              (->> (.value inst id-attrib) int anon-by-id-fn :parse-anon
                   (create-features-fn context)))]
      (-> insts
          (weka/map-merge-instances feature-metas instance-val-fn)
          (weka/remove-attributes [id-key])))))
