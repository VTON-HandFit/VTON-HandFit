

export no_proxy="localhost, 127.0.0.1, ::1"

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

CUDA_VISIBLE_DEVICES=4 python gradio_ootd_inp.py