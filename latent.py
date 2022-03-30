import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from os.path import join
from sklearn.metrics import accuracy_score

class LatentSpaceViewer(tf.keras.callbacks.Callback):
    def __init__(self, model,out_dir):
        super(LatentSpaceViewer, self).__init__()
        self.model = model
        self.pca = PCA()
        self.train_rand = []
        self.test_rand = []
        self.out_dir = out_dir

    def on_epoch_end(self, epoch, logs,train_tup,test_tup=None,query_tup=None,support_tup=None,kmeans_seed=None):
        ### Train Data
        train_data_p,train_labels_p,train_subj_ids = train_tup
        print(train_tup)


        if query_tup is not None and support_tup is not None:
            query_data,query_lbls = query_tup
            support_data,support_lbls = support_tup
            comb_data = np.concatenate([train_data_p,query_data,support_data])
            comb_labels = np.concatenate([train_labels_p,query_lbls,support_lbls])
        else:
            comb_data = train_data_p
            comb_labels = train_labels_p

        if comb_labels is None:
            comb_labels = np.ones(comb_data.shape[0])
        lt_sp_train = self.model.encoder(comb_data)
        print(lt_sp_train)
        lt_pca_proj = self.pca.fit_transform(lt_sp_train)

        train_embeds = lt_pca_proj[:train_data_p.shape[0]]
        if query_tup is not None:
            query_embeds = lt_pca_proj[train_data_p.shape[0]:train_data_p.shape[0]+query_data.shape[0]]
        if support_tup is not None:
            support_embeds = lt_pca_proj[train_data_p.shape[0]+query_data.shape[0]:]

        fig,ax = plt.subplots(figsize=(18, 7),ncols=2)
        ax[0].set_title("Latent Space with Ground Truth")
        #per_lbl_means = lt_sp_train.numpy().reshape(8,15,-1).mean(axis=1)
        #per_lbl_pca_proj = self.pca.transform(per_lbl_means)
        cmap = plt.get_cmap("tab10")
        for i,unq in enumerate(np.unique(comb_labels)):
            # Print unsupervised embeddings
            ax[0].scatter(train_embeds[train_labels_p == unq,0], train_embeds[train_labels_p == unq, 1],label=("Class %d"%unq),color=cmap(i))
            # Print query points
            if query_tup is not None:
                ax[0].scatter(query_embeds[query_lbls == unq,0],query_embeds[query_lbls == unq,1],label=("Class %d Query" % unq),color=cmap(i),marker="D",s=125,edgecolors="black",zorder=3)
            if support_tup is not None:
                ax[0].scatter(support_embeds[support_lbls == unq,0],support_embeds[support_lbls == unq,1],label=("Class %d Support" % unq),color=cmap(i),marker="s",s=125,edgecolors="black",zorder=3)
            # Print query points
            #ax[0].scatter(per_lbl_pca_proj[i,0],per_lbl_pca_proj[i,1],marker="D",s=120,color=cmap(i))
        #ax[0].legend()
        # Kmeans
        km_labels = KMeans(np.unique(comb_labels).shape[0],random_state=kmeans_seed).fit_predict(lt_sp_train)
        #np.savetxt('KMeans_label.out', km_labels, delimiter=',')  # X is an array
        ax[1].set_title("Latent Space with KMeans Clusters")
        for unq in np.unique(km_labels):
            ax[1].scatter(lt_pca_proj[km_labels == unq,0],lt_pca_proj[km_labels == unq, 1])
        # Get Rand Index
        if train_labels_p is not None:
            train_rand_score = adjusted_rand_score(train_labels_p,km_labels[:train_labels_p.shape[0]])
            self.train_rand.append(train_rand_score)

            # Print
            ax[1].text(0,1,"RI: %f" % train_rand_score,transform=ax[1].transAxes)
        np.savetxt(join(self.out_dir, "prediction/label_epoch_%d.txt" % epoch), accuracy_score(km_labels, train_labels_p))
        from tslearn.clustering import silhouette_score
        np.savetxt(join(self.out_dir, "silhouette/label_epoch_%d.txt" % epoch),
                   silhouette_score(lt_sp_train, km_labels))

        self.savefig = plt.savefig(join(self.out_dir, "figs/latent_epoch_%d_train.png" % epoch))
        if train_subj_ids is not None:
            np.savetxt(join(self.out_dir,"labels/labels_epoch_%d.csv"%epoch),np.stack([train_subj_ids,km_labels[:train_data_p.shape[0]]],axis=-1),fmt=["%s","%s"])
        plt.close()
        if test_tup is not None:
            test_data_p,test_labels_p = test_tup
            ### TEST DATA


            lt_sp_test = self.model.get_dense_layer(test_data_p)
            lt_pca_proj_test = self.pca.fit_transform(lt_sp_test)

            fig,ax = plt.subplots(figsize=(18, 7),ncols=2)
            ax[0].set_title("Latent Space with Ground Truth")
            for unq in np.unique(test_labels_p):
                ax[0].scatter(lt_pca_proj_test[test_labels_p == unq, 0], lt_pca_proj_test[test_labels_p == unq, 1])

            # Kmeans
            km_labels_test = KMeans(np.unique(test_labels_p).shape[0],random_state=kmeans_seed).fit_predict(lt_sp_test)
            ax[1].set_title("Latent Space with KMeans Clusters")
            for unq in np.unique(km_labels_test):
                ax[1].scatter(lt_pca_proj_test[km_labels_test == unq,0],lt_pca_proj_test[km_labels_test == unq,1])
            # Get Rand Index
            test_rand_score = adjusted_rand_score(test_labels_p,km_labels_test)
            self.test_rand.append(test_rand_score)

            # Print
            ax[1].text(0,1,"RI: %f" % test_rand_score,transform=ax[1].transAxes)
            plt.savefig(join(self.out_dir,"figs/latent_epoch_%d_test.png"%epoch))
            plt.close()