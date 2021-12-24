export const roomName = JSON.parse(document.getElementById('room-name').textContent);

export function update_yaml_configurations(e) {
  const data = JSON.parse(e.data);
  for (const [stage, stage_data] of Object.entries(data)) {
    for (const [attr, value] of Object.entries(stage_data)) {
      const id = '#' + stage + '_' + attr;
      document.querySelector(id).value = value;
    }
  }
}

export function create_socket(sub_path) {
  const newSocket = new WebSocket(
    'ws://'
    + window.location.host
    + '/ws/web/'
    + sub_path
    + roomName
    + '/'
  );
  return newSocket;
}

export function update_console(str) {
  document.querySelector('#chat-log').value += (str + '\n');
}

export function update_console_v2(str) {
  document.querySelector('#chat-log').value += (str + '\n');
  let s = document.querySelector('#chat-log').value;
  let ind = s.slice(0,-1).lastIndexOf('\n');
  document.querySelector('#chat-log').value = s.slice(0,ind+1) + str + '\n';

}

export const yamlUpdateSocket = create_socket("yaml/update/");

yamlUpdateSocket.onmessage = function(e) {
  update_yaml_configurations(e);
  update_console("yaml parameters updated.");
};
