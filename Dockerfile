FROM ubuntu:16.06


RUN apt-get update
RUN apt-get install -y software-properties-common
RUN apt-get install -y make
RUN apt-get install -y libexpat1-dev
RUN add-apt-repository ppa:avsm/ppa
RUN apt-get update
RUN apt-get install -y ocaml-4.02\* ocaml-native-compilers camlp4-extra opam
RUN opam init
RUN eval `opam config env`
RUN . /root/.opam/opam-init/init.sh > /dev/null 2> /dev/null || true
RUN touch ~/.ocamlinit
RUN echo 'let () = try Topdirs.dir_directory (Sys.getenv "OCAML_TOPLEVEL_PATH") with Not_found -> ();;' >> ~/.ocamlinit
RUN apt-get install -y m4
RUN opam depext ocaml-expat.0.9.1
RUN opam install libra-tk
RUN apt-get purge -y ocaml
RUN apt-get install -y ocaml=4.02\*
RUN apt-get install -y python3-pyqt5
RUN apt-get install -y python-sip
RUN apt-get install -y python-pip
RUN apt-get install -y python-numpy
RUN apt-get install -y python-scipy
RUN apt-get install -y build-essential python-dev
RUN pip install scikit-learn[alldeps]
RUN pip install pandas
RUN pip install Cython
RUN apt-get install -y libfreetype6-dev libxft-dev
RUN pip install pypng
RUN sudo pip install scikit-image

RUN export uid=1000 gid=1000 && \
    mkdir -p /home/developer && \
    echo "developer:x:${uid}:${gid}:Developer,,,:/home/developer:/bin/bash" >> /etc/passwd && \
    echo "developer:x:${uid}:" >> /etc/group && \
    echo "developer ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/developer && \
    chmod 0440 /etc/sudoers.d/developer && \
    chown ${uid}:${gid} -R /home/developer

RUN export QT_X11_NO_MITSHM=1

USER developer
ENV HOME /home/developer

ADD . .

WORKDIR .
CMD ["python", "MediForUI.py"] 
