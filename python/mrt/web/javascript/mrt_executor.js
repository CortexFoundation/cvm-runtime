import { roomName } from './utils.js';

const mrtExecuteSocket = new WebSocket(
  'ws://'
  + window.location.host
  + '/ws/web/mrt/execute/'
  + roomName
  + '/'
);

const activation_flag = "___activate_executor___";
const mrtExecutor = document.querySelector('#mrt-executor');

mrtExecuteSocket.onmessage = function(e) {
  const data = JSON.parse(e.data);
  if (data.message ===  activation_flag) {
    mrtExecutor.disabled = false;
  } else {
    document.querySelector('#chat-log').value += (data.message + '\n');
  }
};

mrtExecuteSocket.onclose = function(e) {
  console.error('Chat socket closed unexpectedly');
};

const yamlCollectSocket = new WebSocket(
  'ws://'
  + window.location.host
  + '/ws/web/yaml/collect/'
  + roomName
  + '/'
);

yamlCollectSocket.onmessage = function(e) {
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
  mrtExecuteSocket.send(JSON.stringify(dict));
};

mrtExecutor.onclick = function(e) {
  mrtExecutor.disabled = true;
  yamlCollectSocket.send(null);
};
