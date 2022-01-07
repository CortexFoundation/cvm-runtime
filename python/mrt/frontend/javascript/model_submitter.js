import { roomName, create_socket, update_console_v2 } from './utils.js';

const modelSubmitSocket = create_socket("model/submit/");

const mrtExecutor = document.querySelector('#mrt-executor');
const modelSubmitter = document.querySelector('#model-submitter');

modelSubmitSocket.onmessage = function(e) {
  const data = JSON.parse(e.data);
  if ('activate' in data) {
    mrtExecutor.disabled = false;
    modelSubmitter.disabled = false;
  }
  if ('message' in data) {
    if ('first' in data) {
      update_console(data.message);
    } else {
      update_console_v2(data.message);
    }
  }
};

modelSubmitSocket.onclose = function(e) {
  console.error('model submit socket closed unexpectedly');
};

modelSubmitter.onclick = function(e) {
  mrtExecutor.disabled = true;
  modelSubmitter.disabled = true;
  let text_data = new Object();
  text_data['symbol'] = document.querySelector('#symbol-locator').value;
  text_data['params'] = document.querySelector('#params-locator').value;
  text_data['dst'] = document.querySelector('#COMMON_MODEL_DIR').value;
  text_data['host'] = document.querySelector('#host-locator').value;
  modelSubmitSocket.send(JSON.stringify(text_data));
};
