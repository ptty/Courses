# Persistant Volumes Setup

## 1. Ordering volume & FS setup
    TBD

### 1.1 Ordering
    <TBD>
### 1.2 Setup FS
    <TBD>
### 1.3 Mounting
    <TBD>

## 2. Setup **kubectl** client on the host
    //TODO: Shell be ref. to indivdual instruction


### 2.1 RHEL
    <TBD> 

#### 2.2 Setup alias
    echo alias k=\'~/DOCKER/kubectl\' >> ~/.profile && source ~/.profile



### 2.2. Connect to cluster
<TBD>

## 3. Setup Persistent Storage


    cat > storage-class.yml << EOF 
    kind: StorageClass
    apiVersion: storage.k8s.io/v1
        metadata:
          name: $local-storage-name
        provisioner: $provisioner
        reclaimPolicy: $policy
        mountOptions:
         - debug
        volumeBindingMode: $binding-mode
    EOF


**Parameters:**

* **metadata.name** - Storage class aka storage profile name 

* **$provisioner** - for exisiting local mountpoint use "kubernetes.io/no-provisioner" (ref. https://kubernetes.io/docs/concepts/storage/storage-classes/#provisioner)

* **$policy** - 

* **$volumeBindingMode** -  







