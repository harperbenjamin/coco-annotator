from flask_restplus import Namespace, Resource, reqparse
from flask_login import login_required, current_user
from werkzeug.datastructures import FileStorage
from flask import send_file

from ..util import query_util, coco_util
from database import ImageModel, DatasetModel, AnnotationModel
import tensorflow as tf
import tensorflow_hub as hub

from PIL import Image
import datetime
import os
import io


api = Namespace("image", description="Image related operations")


image_all = reqparse.RequestParser()
image_all.add_argument("fields", required=False, type=str)
image_all.add_argument("page", default=1, type=int)
image_all.add_argument("per_page", default=50, type=int, required=False)

image_upload = reqparse.RequestParser()
image_upload.add_argument(
    "image", location="files", type=FileStorage, required=True, help="PNG or JPG file"
)
image_upload.add_argument(
    "dataset_id", required=True, type=int, help="Id of dataset to insert image into"
)

image_download = reqparse.RequestParser()
image_download.add_argument("asAttachment", type=bool, default=False)
image_download.add_argument("thumbnail", type=bool, default=False)
image_download.add_argument("width", type=int)
image_download.add_argument("height", type=int)

copy_annotations = reqparse.RequestParser()
copy_annotations.add_argument(
    "category_ids",
    location="json",
    type=list,
    required=False,
    default=None,
    help="Categories to copy",
)

thunder = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")


def _movenet(input_image):
    """Runs detection on an input image.

    Args:
      input_image: A [1, height, width, 3] tensor represents the input image
        pixels. Note that the height/width should already be resized and match the
        expected input resolution of the model before passing into this function.

    Returns:
      A [1, 1, 17, 3] float numpy array representing the predicted keypoint
      coordinates and scores.
    """
    module = thunder.signatures["serving_default"]

    # SavedModel format expects tensor type of int32.
    input_image = tf.cast(input_image, dtype=tf.int32)
    # Run model inference.
    outputs = model(input_image)
    # Output is a [1, 1, 17, 3] tensor.
    keypoint_with_scores = outputs["output_0"].numpy()
    return keypoint_with_scores


@api.route("/")
class Images(Resource):
    @api.expect(image_all)
    @login_required
    def get(self):
        """ Returns all images """
        args = image_all.parse_args()
        per_page = args["per_page"]
        page = args["page"] - 1
        fields = args.get("fields", "")

        images = current_user.images.filter(deleted=False)
        total = images.count()
        pages = int(total / per_page) + 1

        images = images.skip(page * per_page).limit(per_page)
        if fields:
            images = images.only(*fields.split(","))

        return {
            "total": total,
            "pages": pages,
            "page": page,
            "fields": fields,
            "per_page": per_page,
            "images": query_util.fix_ids(images.all()),
        }

    @api.expect(image_upload)
    @login_required
    def post(self):
        """ Creates an image """
        args = image_upload.parse_args()
        image = args["image"]
        reformat = args.get("reformat")

        dataset_id = args["dataset_id"]
        try:
            dataset = DatasetModel.objects.get(id=dataset_id)
        except:
            return {"message": "dataset does not exist"}, 400
        directory = dataset.directory
        path = os.path.join(directory, image.filename)

        if os.path.exists(path):
            return {"message": "file already exists"}, 400

        # Optional argument to resize the image to a standard size
        if reformat:
            tf_image = tf.image.decode_jpeg(io.BytesIO(image.read()))
            tf_image = tf.image.resize_with_pad(tf_image, reformat["x"], reformat["y"])
            _predict_points(tf_image)
            pil_image = Image.fromarray(tf_image)
        else:
            pil_image = Image.open(io.BytesIO(image.read()))

        pil_image.save(path)

        image.close()
        pil_image.close()
        db_image = ImageModel.create_from_path(path, dataset_id).save()
        return db_image.id


def _predict_points(img):
    keypoint_with_scores = _movenet(input_image)

    # TODO iterate over keypoints and add them if they meet a certain confidence threshold
    # TODO add a db migration so that the skeleton ??? is always in the DB
    # TODO store in the DB associated with the image


@api.route("/<int:image_id>")
class ImageId(Resource):
    @api.expect(image_download)
    @login_required
    def get(self, image_id):
        """ Returns category by ID """
        args = image_download.parse_args()
        as_attachment = args.get("asAttachment")
        thumbnail = args.get("thumbnail")

        image = current_user.images.filter(id=image_id, deleted=False).first()

        if image is None:
            return {"success": False}, 400

        width = args.get("width")
        height = args.get("height")

        if not width:
            width = image.width
        if not height:
            height = image.height

        pil_image = image.open_thumbnail() if thumbnail else Image.open(image.path)

        pil_image.thumbnail((width, height), Image.ANTIALIAS)
        image_io = io.BytesIO()
        pil_image = pil_image.convert("RGB")
        pil_image.save(image_io, "JPEG", quality=90)
        image_io.seek(0)

        return send_file(
            image_io, attachment_filename=image.file_name, as_attachment=as_attachment
        )

    @login_required
    def delete(self, image_id):
        """ Deletes an image by ID """
        image = current_user.images.filter(id=image_id, deleted=False).first()
        if image is None:
            return {"message": "Invalid image id"}, 400

        if not current_user.can_delete(image):
            return {"message": "You do not have permission to download the image"}, 403

        image.update(set__deleted=True, set__deleted_date=datetime.datetime.now())
        return {"success": True}


@api.route("/copy/<int:from_id>/<int:to_id>/annotations")
class ImageCopyAnnotations(Resource):
    @api.expect(copy_annotations)
    @login_required
    def post(self, from_id, to_id):
        args = copy_annotations.parse_args()
        category_ids = args.get("category_ids")

        image_from = current_user.images.filter(id=from_id).first()
        image_to = current_user.images.filter(id=to_id).first()

        if image_from is None or image_to is None:
            return {"success": False, "message": "Invalid image ids"}, 400

        if image_from == image_to:
            return {"success": False, "message": "Cannot copy self"}, 400

        if image_from.width != image_to.width or image_from.height != image_to.height:
            return {"success": False, "message": "Image sizes do not match"}, 400

        if category_ids is None:
            category_ids = (
                DatasetModel.objects(id=image_from.dataset_id).first().categories
            )

        query = AnnotationModel.objects(
            image_id=image_from.id, category_id__in=category_ids, deleted=False
        )

        return {"annotations_created": image_to.copy_annotations(query)}


@api.route("/<int:image_id>/coco")
class ImageCoco(Resource):
    @login_required
    def get(self, image_id):
        """ Returns coco of image and annotations """
        image = current_user.images.filter(id=image_id).exclude("deleted_date").first()

        if image is None:
            return {"message": "Invalid image ID"}, 400

        if not current_user.can_download(image):
            return (
                {
                    "message": "You do not have permission to download the images's annotations"
                },
                403,
            )

        return coco_util.get_image_coco(image_id)
