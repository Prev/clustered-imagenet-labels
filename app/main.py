import collections
import os
import json
from datetime import datetime
from flask import Flask, render_template, request, redirect

import utils

app = Flask(__name__)
data = {
    'classids': utils.load_json('data/classids.json'),
    'classname_candiates': utils.load_json('data/new_classname_candiates.json'),
    'class_search_numbers': utils.load_json('data/class_search_numbers.json'),
    'image_urls': utils.load_json('data/image_urls.json'),
    'descendants': utils.load_json('data/descendants.json'),
    'new_labels': utils.load_json('data/new_labels.json'),
    'new_labels_v3': utils.load_json('data/new_labels_v3.json'),
}
blacklists = ('animal', 'device', 'instrument', 'clothing', 'plant part',
              'fruit', 'natural object', 'musical instrument', 'organism', 'vertebrate',
              'course', 'implement', 'support', 'vegetable', 'mammal', 'carnivore', 'placental',
              'furniture', 'equipment', 'electronic equipment', 'arthropod', 'invertebrate',
              'consumer goods', 'nutriment', 'dish', 'plant organ', 'reproductive structure',
              'edible fruit')
logger = utils.get_logger()


@app.route('/')
def main():
    return render_template('main.html', data=data)


@app.route('/<int:image_id>', methods=['GET'])
def annotate_image(image_id):
    classid = data['classids'][image_id]

    candidates = data['classname_candiates'][classid]
    numbers = data['class_search_numbers']

    # main_candidate = candidates[-1]
    # for classname in candidates:
    #     if classname not in blacklists and \
    #         numbers[classname] >= numbers[main_candidate]:
    #         main_candidate = classname
    main_candidate = data['new_labels_v3'][classid]
    original_label = candidates[-1]
    if main_candidate not in candidates:
        candidates.append(main_candidate)

    candidates_info = []
    for classname in candidates:
        descendants = []

        if classname in data['descendants']:
            for d_classid, d_classname in data['descendants'][classname][:10]:
                if d_classid == classid:
                    continue
                descendants.append({
                    'id': d_classid,
                    'name': d_classname,
                    'image_urls': data['image_urls'][d_classid],
                })

        candidates_info.append({
            'name': classname,
            'selected': classname == main_candidate,
            'descendants': descendants,
            'search_number': numbers[classname],
            'is_original_label': classname == original_label,
            'is_new_label': classname == main_candidate,
        })

    return render_template('annotate.html',
        request=request,
        image_id=image_id,
        classid=classid,
        image_urls=data['image_urls'][classid],
        candidates_info=candidates_info,
        selected_candidate=main_candidate,
    )


@app.route('/<int:image_id>', methods=['POST'])
def record(image_id):
    user = request.form.get('u', '')
    result = json.dumps(request.form)
    logger.info(datetime.now().isoformat() + '\t' + request.remote_addr + '\t' + str(image_id) + '\t' + result)
    redirect_url = f'/{(image_id + 1)}?u={user}'
    return redirect(redirect_url)


@app.route('/c/<classid>', methods=['GET'])
def annotate_by_classid(classid):
    image_id = data['classids'].index(classid)
    return redirect(f'/{image_id}')


@app.route('/look-at-once', methods=['GET'])
def look_at_once():
    grouped_classes = collections.defaultdict(list)
    i = 0
    for classid, new_label in data['new_labels_v3'].items():
        grouped_classes[new_label].append({
            'classid': classid,
            'classname': data['classname_candiates'][classid][-1],
            'image_urls': data['image_urls'][classid],
        })
        i += 1
    grouped_classes = dict(grouped_classes)
    print(i)
    return render_template('look_at_once.html', grouped_classes=grouped_classes)


@app.template_filter()
def readable_number(number):
    if number > 1000000000:
        return str(round(number / 1000000000, 1)) + 'B'
    elif number > 1000000:
        return str(round(number / 1000000, 1)) + 'M'
    elif number > 1000:
        return str(round(number / 1000, 1)) + 'K'
    else:
        return str(number)


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        debug=True,
        port=os.environ.get('APP_PORT', 8080)
    )
