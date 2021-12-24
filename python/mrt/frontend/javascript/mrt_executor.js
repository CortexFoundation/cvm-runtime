import { roomName, create_socket, update_console } from './utils.js';

const mrtExecuteSocket = create_socket("mrt/execute/");

const mrtExecutor = document.querySelector('#mrt-executor');
const modelSubmitter = document.querySelector('#model-submitter');

mrtExecuteSocket.onmessage = function(e) {
  const data = JSON.parse(e.data);
  if ('activate' in data) {
    mrtExecutor.disabled = false;
    modelSubmitter.disabled = false;
  }
  if ('message' in data) {
    update_console(data.message);
  }
};

mrtExecuteSocket.onclose = function(e) {
  console.error('mrt execute socket closed unexpectedly');
};

const ConfigWrapperSocket = create_socket("config/wrapper/");

ConfigWrapperSocket.onmessage = function(e) {
  const data = JSON.parse(e.data);
  let dict = new Object();
  for (const [stage, stage_data] of Object.entries(data)) {
    let subdict = new Object();
    for (const attr of Object.keys(stage_data)) {
      const id = '#' + stage + '_' + attr;
      let value = document.querySelector(id).value;
      subdict[attr] = value;
    }
    dict[stage] = subdict;
  }
  // overide pass_name
  const pass_name = document.querySelector('#mrt-stage-selector').value;
  dict["COMMON"]["PASS_NAME"] = pass_name;
  let text_data = new Object();
  text_data['yaml'] = dict;
  text_data['host'] = document.querySelector('#host-locator').value;
  mrtExecuteSocket.send(JSON.stringify(text_data));
};

mrtExecutor.onclick = function(e) {
  mrtExecutor.disabled = true;
  modelSubmitter.disabled = true;
  ConfigWrapperSocket.send(null);
};
